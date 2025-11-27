from absl import app
from etils import eapp

from legal_rag.indexer import ScriptConfig, build_index


def main(argv=None):
    eapp.better_logging()
    app.run(build_index, flags_parser=eapp.make_flags_parser(ScriptConfig), argv=argv)


if __name__ == "__main__":
    main()
