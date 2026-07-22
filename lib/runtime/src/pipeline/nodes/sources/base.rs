// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use crate::engine::AsyncEngineContextProvider;

use super::*;

impl<In: PipelineIO, Out: PipelineIO> Default for Frontend<In, Out> {
    fn default() -> Self {
        Self {
            edge: OnceLock::new(),
            sinks: Arc::new(Mutex::new(HashMap::new())),
        }
    }
}

#[async_trait]
impl<In: PipelineIO, Out: PipelineIO> Source<In> for Frontend<In, Out> {
    async fn on_next(&self, data: In, _: private::Token) -> Result<(), Error> {
        self.edge
            .get()
            .ok_or(PipelineError::NoEdge)?
            .write(data)
            .await
    }

    fn set_edge(&self, edge: Edge<In>, _: private::Token) -> Result<(), PipelineError> {
        self.edge
            .set(edge)
            .map_err(|_| PipelineError::EdgeAlreadySet)?;
        Ok(())
    }
}

#[async_trait]
impl<In: PipelineIO, Out: PipelineIO + AsyncEngineContextProvider> Sink<Out> for Frontend<In, Out> {
    async fn on_data(&self, data: Out, _: private::Token) -> Result<(), Error> {
        let ctx = data.context();

        let mut sinks = self.sinks.lock().unwrap();
        let tx = sinks
            .remove(ctx.id())
            .ok_or(PipelineError::DetachedStreamReceiver)
            .inspect_err(|_| {
                ctx.stop_generating();
            })?;
        drop(sinks);

        Ok(tx
            .send(data)
            .map_err(|_| PipelineError::DetachedStreamReceiver)
            .inspect_err(|_| {
                ctx.stop_generating();
            })?)
    }
}

/// Removes a request's `sinks` entry unless `on_data` consumed it first.
/// `generate` can exit with the entry still in the map — `on_next` failing
/// before dispatch, or the future being dropped at an await point when the
/// client disconnects — and each such exit would otherwise leak the entry
/// permanently. A late `on_data` after removal takes the existing
/// `DetachedStreamReceiver` / `stop_generating` path.
struct SinkEntryGuard<Out> {
    sinks: Arc<Mutex<HashMap<String, oneshot::Sender<Out>>>>,
    id: String,
}

impl<Out> Drop for SinkEntryGuard<Out> {
    fn drop(&mut self) {
        // Skip a poisoned lock rather than double-panic in drop.
        if let Ok(mut sinks) = self.sinks.lock() {
            sinks.remove(&self.id);
        }
    }
}

#[async_trait]
impl<In: PipelineIO + Sync, Out: PipelineIO> AsyncEngine<In, Out, Error> for Frontend<In, Out> {
    async fn generate(&self, request: In) -> Result<Out, Error> {
        let id = request.id().to_string();
        let (tx, rx) = oneshot::channel::<Out>();
        {
            let mut sinks = self.sinks.lock().unwrap();
            sinks.insert(id.clone(), tx);
        }
        let _guard = SinkEntryGuard {
            sinks: self.sinks.clone(),
            id,
        };
        self.on_next(request, private::Token {}).await?;
        Ok(rx.await.map_err(|_| PipelineError::DetachedStreamSender)?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pipeline::{ManyOut, SingleIn, error::PipelineErrorExt};

    #[tokio::test]
    async fn test_frontend_no_edge() {
        let source = Frontend::<SingleIn<()>, ManyOut<()>>::default();
        let error = source
            .generate(().into())
            .await
            .unwrap_err()
            .try_into_pipeline_error()
            .unwrap();

        match error {
            PipelineError::NoEdge => (),
            _ => panic!("Expected NoEdge error"),
        }

        let result = source
            .on_next(().into(), private::Token)
            .await
            .unwrap_err()
            .try_into_pipeline_error()
            .unwrap();

        match result {
            PipelineError::NoEdge => (),
            _ => panic!("Expected NoEdge error"),
        }
    }

    #[tokio::test]
    async fn test_generate_error_removes_sink_entry() {
        // No edge is set, so on_next fails after the sink entry is inserted.
        let source = Frontend::<SingleIn<()>, ManyOut<()>>::default();
        source.generate(().into()).await.unwrap_err();
        assert_eq!(source.sinks.lock().unwrap().len(), 0);
    }

    #[tokio::test]
    async fn test_generate_cancellation_removes_sink_entry() {
        use futures::FutureExt;

        // Forward sink that accepts the request and never responds, parking
        // generate at the rx.await point.
        struct NullSink;

        #[async_trait]
        impl Sink<SingleIn<()>> for NullSink {
            async fn on_data(&self, _: SingleIn<()>, _: private::Token) -> Result<(), Error> {
                Ok(())
            }
        }

        let source = Arc::new(Frontend::<SingleIn<()>, ManyOut<()>>::default());
        source.link(Arc::new(NullSink)).unwrap();

        // Single poll: inserts the entry, dispatches, parks on rx — then the
        // future is dropped, emulating a client disconnect.
        assert!(source.generate(().into()).now_or_never().is_none());
        assert_eq!(source.sinks.lock().unwrap().len(), 0);
    }
}
