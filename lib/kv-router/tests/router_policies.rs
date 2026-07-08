// SPDX-FileCopyrightText: Copyright (c) 2026 DeepInfra. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Validates the router policy profiles baked into the runtime images
//! (deploy/router-policies/) against the real parser, so a bad edit fails
//! CI instead of crash-looping frontends at startup.

use dynamo_kv_router::scheduling::RouterPolicyConfig;

#[test]
fn default_policy_parses_with_expected_classes() {
    let yaml = include_str!("../../../deploy/router-policies/default.yaml");
    let config = RouterPolicyConfig::from_yaml(yaml).expect("default router policy must parse");
    let profile = config.resolve_profile(Some("any/model"), None, Default::default());
    let names: Vec<_> = profile.classes().iter().map(|c| c.name.as_str()).collect();
    assert_eq!(names, vec!["standard", "no-queue"]);

    let no_queue = &profile.classes()[1];
    assert_eq!(no_queue.request_queue_limit_per_worker, Some(0));
    assert_eq!(
        profile.direct_class_index(Some("no-queue")),
        Some(1),
        "no-queue must be an explicit header-selectable class"
    );
    assert_eq!(
        profile.resolve_class_index(Some("unknown-class"), 100),
        0,
        "unknown class names must fall back to the default family"
    );
}
