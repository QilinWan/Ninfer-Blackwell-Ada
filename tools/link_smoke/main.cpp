// Link smoke: references the public engine surface so the archives have to close under a real
// executable link. Static archives accept unresolved symbols, which is exactly how a dropped
// kernel route would slip past a library-only build; this target turns that into a link error.
//
// It never constructs an Engine, so it runs on a machine without a GPU:
//     cmake --build <dir> --target ninfer-link-smoke && ./tools/link_smoke/ninfer-link-smoke
#include <ninfer/engine.h>

#include <cstdio>

namespace {

// Engine member functions live in the engine archive and reach the model registry, the Op
// dispatchers and the artifact reader through their own references.
using EngineMember = const ninfer::EngineOptions& (ninfer::Engine::*)() const;
using LoadMember   = ninfer::LoadSummary (ninfer::Engine::*)() const;

} // namespace

int main(int argc, char** argv) {
    ninfer::EngineOptions options;
    if (argc > 1) { options.artifact_path = argv[1]; }

    static const EngineMember options_member = &ninfer::Engine::options;
    static const LoadMember load_member      = &ninfer::Engine::load_summary;
    static const auto tokenizer              = &ninfer::Engine::count_tokens;

    std::printf("ninfer: engine archives linked (options=%p load=%p count=%p, artifact=%s)\n",
                static_cast<const void*>(&options_member), static_cast<const void*>(&load_member),
                static_cast<const void*>(&tokenizer),
                options.artifact_path.empty() ? "<unset>" : options.artifact_path.string().c_str());
    return 0;
}
