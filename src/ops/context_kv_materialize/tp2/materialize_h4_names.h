// Renames that turn upstream's context_kv_materialize/materialize.cu into the two-device four-KV-head
// copy when materialize_shard_h4.cu includes it: the KV-head count (an overridable constant of the
// source) and the entry point. Include it once, right before the upstream source.

#define NINFER_CONTEXT_KV_MATERIALIZE_KV_HEADS 4
#define context_kv_materialize_launch context_kv_materialize_h4_launch
