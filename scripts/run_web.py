"""Serve the built local application; credentials remain on the Python server."""
import argparse
import os
from pathlib import Path

import uvicorn


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8790)
    parser.add_argument("--data-dir", type=Path)
    args = parser.parse_args()
    if args.data_dir:
        os.environ["MEDICAL_COPILOT_DATA_DIR"] = str(args.data_dir.resolve())
    from medical_copilot.web.app import create_app
    uvicorn.run(create_app(), host="127.0.0.1", port=args.port, access_log=False)


if __name__ == "__main__":
    main()
