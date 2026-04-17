from src.app_bootstrap import run_app


def main() -> None:
    run_app("AI Yüz Tanıma Demo", age_bias_years=-35.0)


if __name__ == "__main__":
    main()
