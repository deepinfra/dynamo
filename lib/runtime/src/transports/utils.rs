// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::{future::Future, sync::Arc};

use anyhow::Result;
use tokio::task::JoinHandle;

/// Handle to a tokio runtime that lives on a dedicated OS thread.
///
/// The runtime is created and owned on a dedicated OS thread so it can be
/// dropped safely: dropping a multi-threaded runtime blocks while its worker
/// threads are joined, which panics if attempted from within an async context.
///
/// Cloning a `RuntimeHandle` shares the same underlying runtime. When the last
/// clone is dropped the dedicated OS thread wakes, drops the runtime, and exits,
/// reclaiming its worker threads. This is what previously leaked: the dedicated
/// thread used to park on `pending::<()>()` forever, so the runtime (and its
/// worker threads) could never be reclaimed even after every caller was gone.
#[derive(Clone)]
pub struct RuntimeHandle {
    inner: Arc<RuntimeInner>,
}

struct RuntimeInner {
    runtime: Arc<tokio::runtime::Runtime>,
    // Dropping this sender (which happens when the last `RuntimeHandle` clone is
    // dropped) signals the dedicated OS thread to stop parking and shut the
    // runtime down.
    _shutdown: tokio::sync::oneshot::Sender<()>,
}

impl RuntimeHandle {
    /// Spawn a future onto the runtime.
    pub fn spawn<F>(&self, future: F) -> JoinHandle<F::Output>
    where
        F: Future + Send + 'static,
        F::Output: Send + 'static,
    {
        self.inner.runtime.spawn(future)
    }

    /// Access the underlying tokio runtime handle.
    pub fn handle(&self) -> &tokio::runtime::Handle {
        self.inner.runtime.handle()
    }
}

pub async fn build_in_runtime<
    T: Send + Sync + 'static,
    F: Future<Output = Result<T>> + Send + 'static,
>(
    f: F,
    num_threads: usize,
) -> Result<(T, RuntimeHandle)> {
    let (tx, rx) = tokio::sync::oneshot::channel();
    let (shutdown_tx, shutdown_rx) = tokio::sync::oneshot::channel::<()>();

    let runtime = Arc::new(
        tokio::runtime::Builder::new_multi_thread()
            .worker_threads(num_threads)
            .enable_all()
            .build()?,
    );

    let runtime_clone = runtime.clone();
    std::thread::spawn(move || {
        runtime_clone.block_on(async move {
            let result = f.await;
            tx.send(result)
                .unwrap_or_else(|_| panic!("This should never happen!"));

            // Park until the last `RuntimeHandle` is dropped (dropping its sender
            // resolves this recv with an error), instead of parking forever. This
            // lets the runtime be reclaimed once no caller needs it.
            let _ = shutdown_rx.await;
        });

        // `block_on` has returned, so the shutdown signal fired and every external
        // `RuntimeHandle` is gone — `runtime_clone` is now the last strong
        // reference. Drop it here: on a plain OS thread it is safe to block while
        // the runtime joins its worker threads (unlike dropping in async context).
        drop(runtime_clone);
    });

    let result = rx.await??;

    Ok((
        result,
        RuntimeHandle {
            inner: Arc::new(RuntimeInner {
                runtime,
                _shutdown: shutdown_tx,
            }),
        },
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn thread_count() -> usize {
        std::fs::read_dir("/proc/self/task")
            .map(|d| d.count())
            .unwrap_or(0)
    }

    /// Wait until `pred()` holds or the deadline passes; returns the last value.
    fn wait_until(mut pred: impl FnMut() -> bool) -> bool {
        for _ in 0..100 {
            if pred() {
                return true;
            }
            std::thread::sleep(std::time::Duration::from_millis(50));
        }
        pred()
    }

    #[tokio::test]
    async fn runtime_is_reclaimed_when_handle_dropped() {
        const WORKERS: usize = 4;
        let baseline = thread_count();

        let (val, handle) = build_in_runtime(async { Ok(42u32) }, WORKERS)
            .await
            .expect("build_in_runtime failed");
        assert_eq!(val, 42);

        // The dedicated runtime spawned its worker threads.
        assert!(
            wait_until(|| thread_count() >= baseline + WORKERS),
            "expected at least {WORKERS} new threads, baseline={baseline} now={}",
            thread_count()
        );

        // Dropping the (only) handle must wake the dedicated OS thread, drop the
        // runtime, and reclaim every worker thread — the bug this fix addresses.
        drop(handle);

        assert!(
            wait_until(|| thread_count() <= baseline + 1),
            "threads not reclaimed after drop: baseline={baseline} now={}",
            thread_count()
        );
    }

    #[tokio::test]
    async fn cloned_handle_keeps_runtime_alive_until_last_drop() {
        const WORKERS: usize = 4;
        let baseline = thread_count();

        let (_, handle) = build_in_runtime(async { Ok(()) }, WORKERS)
            .await
            .expect("build_in_runtime failed");
        assert!(wait_until(|| thread_count() >= baseline + WORKERS));

        let clone = handle.clone();
        drop(handle);

        // Runtime must stay up while a clone is alive, and still be usable.
        std::thread::sleep(std::time::Duration::from_millis(150));
        assert!(
            thread_count() >= baseline + WORKERS,
            "runtime reclaimed while a clone was still alive"
        );
        let spawned = clone.spawn(async { 7u32 }).await.expect("spawn failed");
        assert_eq!(spawned, 7);

        drop(clone);
        assert!(
            wait_until(|| thread_count() <= baseline + 1),
            "threads not reclaimed after last clone dropped: baseline={baseline} now={}",
            thread_count()
        );
    }
}
