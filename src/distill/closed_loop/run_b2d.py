"""Stock Bench2Drive evaluator with the server-args patch only."""
import sys


def main() -> int:
    from tritrack import patches
    patches.patch_server_args()
    from leaderboard import leaderboard_evaluator
    try:
        leaderboard_evaluator.main()
    except SystemExit as exc:
        return int(exc.code or 0)
    return 0


if __name__ == "__main__":
    sys.exit(main())
