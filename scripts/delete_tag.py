# -*- coding: utf-8 -*-
"""Delete a manifest (and every tag pointing at it) from Docker Hub.

    python scripts/delete_tag.py --repo zhanghangjie/ebba \
        --digest sha256:... --proxy http://127.0.0.1:7892

Deleting is by digest, so two tags sharing one image are removed together.
This is irreversible: the tag disappears from the repository immediately.
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import push_manifest as p  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--digest", required=True)
    ap.add_argument("--proxy", default=None)
    ap.add_argument("--yes", action="store_true", help="skip the confirmation echo")
    args = ap.parse_args()

    user, secret = p.credentials()
    tokens = p.TokenSource(user, secret, args.repo, args.proxy)
    tokens.get()
    if not args.yes:
        print(f"deleting {args.repo}@{args.digest} as {user}")

    status, body, _ = p.http(
        "DELETE",
        f"{p.REGISTRY}/v2/{args.repo}/manifests/{args.digest}",
        token=tokens.get(),
        proxy=args.proxy,
    )
    print(f"DELETE -> HTTP {status}")
    if status not in (202, 200, 204):
        print(body.decode(errors="replace")[:400])
        return 1

    st, body, _ = p.http(
        "GET", f"{p.REGISTRY}/v2/{args.repo}/tags/list",
        token=tokens.get(), proxy=args.proxy,
    )
    print(f"remaining tags: {body.decode(errors='replace')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
