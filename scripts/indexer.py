from absl import app
from etils import eapp

from legal_rag.indexer import ScriptConfig, build_index


def main(argv=None):
    eapp.better_logging()

    def run_and_print_path(cfg: ScriptConfig):
        index_path = build_index(cfg)
        print(index_path)

    app.run(
        run_and_print_path, flags_parser=eapp.make_flags_parser(ScriptConfig), argv=argv
    )


if __name__ == "__main__":
    main()
