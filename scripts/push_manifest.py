# -*- coding: utf-8 -*-
"""Push an image tag by uploading only its manifest, straight to the registry.

`docker push` hangs on this particular image after every layer has already been
uploaded (the daemon never completes the manifest step).  The blobs are all in
the registry, so the missing piece is one small JSON document: this script
reads the image manifest out of a `docker save` OCI archive and PUTs it with a
token from auth.docker.io.

    python scripts/push_manifest.py --tar release/full-image.tar \
        --repo zhanghangjie/ebba --tag full --platform linux/amd64

Credentials come from the Docker credential helper and are never printed.
"""
from __future__ import annotations

import argparse
import base64
import json
import subprocess
import sys
import tarfile
import time
import urllib.error
import urllib.parse
import urllib.request

AUTH = "https://auth.docker.io/token"
REGISTRY = "https://registry-1.docker.io"
CRED_SERVER = "https://index.docker.io/v1/"


def http(method, url, token=None, body=None, content_type=None, proxy=None,
         attempts=4, extra_headers=None):
    """HTTP with retries: the proxy in front of this host drops TLS handshakes
    intermittently (SSLEOFError / ConnectionReset), and a retry usually wins."""
    headers = dict(extra_headers or {})
    if token:
        headers["Authorization"] = "Bearer " + token
    if content_type:
        headers["Content-Type"] = content_type
    opener = urllib.request.build_opener(
        urllib.request.ProxyHandler({"http": proxy, "https": proxy} if proxy else {})
    )
    last = None
    for attempt in range(1, attempts + 1):
        req = urllib.request.Request(url, data=body, headers=headers, method=method)
        try:
            with opener.open(req, timeout=60) as r:
                return r.status, r.read(), dict(r.headers)
        except urllib.error.HTTPError as e:
            return e.code, e.read(), dict(e.headers)
        except Exception as e:  # SSLError, URLError, ConnectionReset, timeouts
            last = e
            if attempt < attempts:
                print(f"  retry {attempt}/{attempts - 1}: {type(e).__name__}: {e}")
                time.sleep(2 * attempt)
    raise last


def credentials():
    out = subprocess.run(
        ["docker-credential-desktop", "get"],
        input=CRED_SERVER.encode(),
        capture_output=True,
        check=True,
    ).stdout
    data = json.loads(out.decode())
    return data["Username"], data["Secret"]


def get_token(user, secret, repo, proxy=None):
    """Mint a push-scoped token.

    A GET with Basic auth silently downgrades the scope to pull-only; the form
    POST with grant_type=password is what actually returns push rights.
    """
    form = urllib.parse.urlencode({
        "grant_type": "password",
        "username": user,
        "password": secret,
        "service": "registry.docker.io",
        "scope": f"repository:{repo}:push,pull",
        "client_id": "docker",
    }).encode()
    status, body, _ = http(
        "POST", AUTH, token=None, body=form,
        content_type="application/x-www-form-urlencoded", proxy=proxy,
    )
    if status != 200:
        raise SystemExit(f"token request failed: HTTP {status} {body[:300]!r}")
    payload = json.loads(body.decode())
    token = payload.get("access_token") or payload["token"]
    claims = token.split(".")[1]
    claims += "=" * (-len(claims) % 4)
    actions = [
        a.get("actions") for a in
        json.loads(base64.urlsafe_b64decode(claims)).get("access", [])
    ]
    print(f"token actions: {actions}")
    if "push" not in sum((a or [] for a in actions), []):
        raise SystemExit("token has no push scope; refusing to continue")
    return token


def load_manifest(tar_path, platform):
    os_want, arch_want = platform.split("/")[:2]

    def blob(tf, digest):
        return tf.extractfile("blobs/sha256/" + digest.split(":")[1]).read()

    with tarfile.open(tar_path, "r:") as tf:
        index = json.loads(tf.extractfile("index.json").read().decode())
        candidates = []
        for entry in index.get("manifests", []):
            mt = entry.get("mediaType", "")
            if "manifest" in mt and "list" not in mt and "index" not in mt:
                candidates.append(entry)
            elif "index" in mt or "list" in mt:
                sub = json.loads(blob(tf, entry["digest"]))
                for m in sub.get("manifests", []):
                    plat = m.get("platform") or {}
                    if plat.get("os") == "unknown":
                        continue  # attestation
                    if plat.get("os") == os_want and plat.get("architecture") == arch_want:
                        candidates.append(m)

        for entry in candidates:
            raw = blob(tf, entry["digest"])
            man = json.loads(raw.decode())
            refs = [man["config"]] + man["layers"]
            return man, raw, refs, entry["digest"]
    raise SystemExit("no image manifest for the requested platform in the archive")


class TokenSource:
    """Push-scoped tokens live about five minutes; a long upload must renew.

    Refreshing does not disturb an in-flight upload session — the session is
    keyed by its Location URL, not by the token that started it.
    """

    def __init__(self, user, secret, repo, proxy):
        self.user = user
        self.secret = secret
        self.repo = repo
        self.proxy = proxy
        self.token = None
        self.expires_at = 0.0

    def get(self, refresh=False):
        if self.token and not refresh and time.time() < self.expires_at - 60:
            return self.token
        self.token = get_token(self.user, self.secret, self.repo, proxy=self.proxy)
        claims = self.token.split(".")[1]
        claims += "=" * (-len(claims) % 4)
        self.expires_at = json.loads(
            base64.urlsafe_b64decode(claims)
        ).get("exp", time.time() + 240)
        return self.token


def upload_blob(repo, digest, stream, size, tokens, proxy, chunk=8 << 20):
    """Upload one blob, in chunks when it is large.

    A single 3.9 GB PUT is dropped by the proxy mid-request ("write operation
    timed out"), so large blobs go up as a resumable PATCH stream: 8 MB per
    request, each retried, and the server's accepted offset is re-read after a
    failure so the upload resumes instead of restarting.

    `stream` is a file object over the blob bytes (never the whole blob in
    memory) and `size` is its length.
    """
    status, _, headers = http(
        "POST", f"{REGISTRY}/v2/{repo}/blobs/uploads/",
        token=tokens.get(), body=b"", proxy=proxy,
    )
    if status not in (202, 201):
        raise SystemExit(f"upload start failed: HTTP {status}")
    loc = headers.get("Location") or headers["location"]

    def absolute(url):
        return url if url.startswith("http") else REGISTRY + url

    if size <= 2 * chunk:
        status, body, _ = http(
            "PUT",
            absolute(loc) + ("&" if "?" in loc else "?")
            + "digest=" + urllib.parse.quote(digest),
            token=tokens.get(), body=stream.read(),
            content_type="application/octet-stream", proxy=proxy, attempts=3,
        )
        if status in (201, 204):
            return True
        print(f"  direct PUT failed (HTTP {status}); retrying in chunks")
        stream.seek(0)

    offset = 0
    while offset < size:
        stream.seek(offset)
        piece = stream.read(chunk)
        want = len(piece)
        ok = False
        for attempt in range(1, 6):
            try:
                status, body, headers = http(
                    "PATCH", absolute(loc), token=tokens.get(), body=piece,
                    content_type="application/octet-stream", proxy=proxy,
                    attempts=1,
                    extra_headers={
                        "Content-Range": f"{offset}-{offset + want - 1}"
                    },
                )
            except Exception as exc:
                print(f"  chunk @{offset}: {type(exc).__name__}, resyncing")
                status, body, headers = 0, b"", {}
            if status in (202, 204):
                new_loc = headers.get("Location") or headers.get("location")
                if new_loc:
                    loc = new_loc
                offset += want
                ok = True
                break
            if status in (401, 403):
                print(f"  chunk @{offset}: HTTP {status}, renewing token")
                tokens.get(refresh=True)
            else:
                print(f"  chunk @{offset}: HTTP {status} {body[:120]!r}")
            # ask the registry how much it actually stored, then resume there
            try:
                st2, _, h2 = http("GET", absolute(loc), token=tokens.get(),
                                  proxy=proxy, attempts=1)
                done = int(h2.get("Range", "").split("-")[-1] or -1) + 1
                if st2 == 204 and done > offset:
                    offset = done
                    ok = True
                    break
            except Exception:
                pass
            time.sleep(1 + attempt)
        if not ok:
            print(f"  giving up at offset {offset} of {size}")
            return False
        if offset % (chunk * 16) == 0 or offset >= size:
            print(f"  {offset / (1 << 20):8.1f} / {size / (1 << 20):.1f} MB", flush=True)

    status, body, _ = http(
        "PUT",
        absolute(loc) + ("&" if "?" in loc else "?")
        + "digest=" + urllib.parse.quote(digest),
        token=tokens.get(), body=b"", content_type="application/octet-stream",
        proxy=proxy, attempts=4,
    )
    print(f"  finalize -> HTTP {status}")
    return status in (201, 204)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tar", required=True)
    ap.add_argument("--repo", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--platform", default="linux/amd64")
    ap.add_argument("--proxy", default=None, help="e.g. http://127.0.0.1:7892")
    ap.add_argument("--upload-missing", action="store_true",
                    help="upload any referenced blob that the registry lacks")
    args = ap.parse_args()

    man, raw, refs, digest = load_manifest(args.tar, args.platform)
    print(f"manifest {digest}  mediaType={man['mediaType']}  blobs={len(refs)}")

    user, secret = credentials()
    print(f"account: {user}")

    tokens = TokenSource(user, secret, args.repo, args.proxy)
    tokens.get()

    missing = []
    for ref in refs:
        status, body, _ = http(
            "HEAD",
            f"{REGISTRY}/v2/{args.repo}/blobs/{ref['digest']}",
            token=tokens.get(),
            proxy=args.proxy,
        )
        if status != 200:
            missing.append((ref["digest"], status))
    if missing:
        print(f"WARNING: {len(missing)} blob(s) missing in the registry:")
        for d, s in missing[:5]:
            print(f"  {d} -> HTTP {s}")
        if not args.upload_missing:
            print("refusing to push a manifest that references missing blobs "
                  "(pass --upload-missing to send them from the tar)")
            return 2
        with tarfile.open(args.tar, "r:") as tf:
            for digest_m, _s in missing:
                member = tf.getmember("blobs/sha256/" + digest_m.split(":")[1])
                print(f"uploading {digest_m[:19]} "
                      f"({member.size / (1 << 20):.1f} MB)")
                stream = tf.extractfile(member)
                if not upload_blob(args.repo, digest_m, stream, member.size,
                                   tokens, args.proxy):
                    print(f"upload failed for {digest_m}")
                    return 2
    print(f"all {len(refs)} referenced blobs are present")

    status, body, _ = http(
        "PUT",
        f"{REGISTRY}/v2/{args.repo}/manifests/{args.tag}",
        token=tokens.get(),
        body=raw,
        content_type=man["mediaType"],
        proxy=args.proxy,
    )
    print(f"PUT manifest -> HTTP {status}")
    if status not in (200, 201):
        print(body.decode(errors="replace")[:800])
        return 1
    print(body.decode(errors="replace")[:400])
    return 0


if __name__ == "__main__":
    sys.exit(main())
