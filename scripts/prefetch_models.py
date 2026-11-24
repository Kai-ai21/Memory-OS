"""Download every model the application loads, into the image (M12.3).

Built from the application's own adapters rather than from model names, so the
cache lands exactly where they will look for it at runtime: a prefetch that
guessed the layout would leave the first request downloading anyway. Each one
is exercised once, which is the only proof the weights are really there.

Run at image build time; the runtime image then sets `HF_HUB_OFFLINE=1`, so a
missing model is an error at startup rather than a download in front of a user.
"""

from memoryos.adapters.embedding.sentence_transformers import build_embedder
from memoryos.adapters.reranking.cross_encoder import CrossEncoderReranker
from memoryos.config import Settings


def main() -> None:
    settings = Settings(_env_file=None)  # type: ignore[call-arg]
    embedder = build_embedder(settings)
    embedder.embed_query(["warm the cache"])
    embedder.count_tokens("and the tokenizer")
    reranker = CrossEncoderReranker(settings.reranker_model, cache_dir=settings.hf_home)
    reranker.rerank("warm", ["the cross-encoder"])
    print(f"models cached in {settings.hf_home}")


if __name__ == "__main__":
    main()
