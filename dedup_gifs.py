#!/usr/bin/env python3
"""GIF 폴더에서 내용이 같은(동일 영상) 중복 파일을 찾아 macOS 휴지통으로 이동한다.

원리:
  1) MD5로 바이트 단위 완전 동일 파일을 그룹핑
  2) 그룹별 대표 파일 1개씩만 남겨 perceptual hash(dHash/aHash)를 계산
  3) LSH 밴딩으로 후보 쌍을 추린 뒤 해밍 거리로 실제 유사 여부 검증
  4) Union-Find로 같은 영상끼리 클러스터링
  5) 클러스터마다 해상도가 가장 큰 파일을 남기고 나머지를 삭제 대상으로 표시

기본 동작은 dry-run이며, --apply를 줘야 실제로 휴지통 이동이 일어난다.
"""
import argparse
import hashlib
import json
import os
import subprocess
import time

from PIL import Image
import numpy as np


def md5_of(path: str) -> str:
    h = hashlib.md5()
    with open(path, "rb") as fp:
        while True:
            chunk = fp.read(1 << 20)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def scan_md5(folder: str) -> dict:
    """content(md5) -> [filenames]. 파일이 완전히 같으면 같은 그룹으로 묶인다."""
    files = sorted(f for f in os.listdir(folder) if f.lower().endswith(".gif"))
    md5_map: dict[str, list[str]] = {}
    for f in files:
        digest = md5_of(os.path.join(folder, f))
        md5_map.setdefault(digest, []).append(f)
    return md5_map


def dhash(img: Image.Image, size: int = 8) -> int:
    """difference hash: 인접 픽셀 밝기 증감 패턴을 비트로 인코딩. 리사이즈/재인코딩에 강함."""
    img = img.convert("L").resize((size + 1, size), Image.LANCZOS)
    arr = np.asarray(img, dtype=np.int16)
    bits = (arr[:, 1:] > arr[:, :-1]).flatten()
    val = 0
    for b in bits:
        val = (val << 1) | int(b)
    return val


def ahash(img: Image.Image, size: int = 8) -> int:
    """average hash: 평균 밝기보다 밝은 픽셀을 비트로 인코딩."""
    img = img.convert("L").resize((size, size), Image.LANCZOS)
    arr = np.asarray(img, dtype=np.float32)
    bits = (arr > arr.mean()).flatten()
    val = 0
    for b in bits:
        val = (val << 1) | int(b)
    return val


def compute_phash(folder: str, md5_map: dict) -> dict:
    """md5 그룹 대표 파일마다 첫 프레임/중간 프레임 해시를 계산.

    깨졌거나 비정상적으로 큰(decompression-bomb 방어에 걸리는) GIF는 건너뛴다 —
    그런 파일은 md5 기준 완전중복만 처리되고 perceptual 클러스터링에서는 제외된다.
    """
    results = {}
    for digest, flist in md5_map.items():
        rep = sorted(flist)[0]
        path = os.path.join(folder, rep)
        try:
            im = Image.open(path)
            im.seek(0)
            first = im.convert("RGB")
            try:
                n_frames = im.n_frames
            except Exception:
                n_frames = 1
            im.seek(n_frames // 2)
            mid = im.convert("RGB")
            results[digest] = {
                "file": rep,
                "dhash1": dhash(first),
                "ahash1": ahash(first),
                "dhash2": dhash(mid),
                "size": im.size,
                "n_frames": n_frames,
            }
        except Exception as e:
            print(f"  건너뜀 (읽기 실패): {rep} ({e})")
    return results


def hamming(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


def cluster_by_similarity(
    md5_map: dict,
    results: dict,
    thresh_d1: int = 6,
    thresh_a1: int = 6,
    thresh_d2: int = 8,
) -> list[list[str]]:
    """비슷한 해시를 가진 md5 그룹들을 하나의 클러스터(=같은 영상)로 묶는다.

    전수비교(O(n^2))는 느리므로 LSH 밴딩(64비트를 16비트씩 4조각)으로 후보 쌍을
    먼저 추리고, 후보에 대해서만 해밍 거리를 정확히 검증한다.
    """
    digests = list(results.keys())
    n = len(digests)
    parent = list(range(n))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    def band_buckets(hash_key, band_bits=16, n_bands=4):
        buckets: dict = {}
        mask = (1 << band_bits) - 1
        for i, d in enumerate(digests):
            h = results[d][hash_key]
            for b in range(n_bands):
                key = (hash_key, b, (h >> (b * band_bits)) & mask)
                buckets.setdefault(key, []).append(i)
        return buckets

    candidates = set()
    for hash_key in ("dhash1", "ahash1"):
        for idxs in band_buckets(hash_key).values():
            if 1 < len(idxs) < 500:  # 너무 흔한 버킷은 무의미하므로 제외
                for i in range(len(idxs)):
                    for j in range(i + 1, len(idxs)):
                        candidates.add((min(idxs[i], idxs[j]), max(idxs[i], idxs[j])))

    for a, b in candidates:
        da, db = results[digests[a]], results[digests[b]]
        if (
            hamming(da["dhash1"], db["dhash1"]) <= thresh_d1
            and hamming(da["ahash1"], db["ahash1"]) <= thresh_a1
            and hamming(da["dhash2"], db["dhash2"]) <= thresh_d2
        ):
            union(a, b)

    groups: dict = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(digests[i])

    clusters = []
    for idxs in groups.values():
        if len(idxs) > 1:
            files = []
            for digest in idxs:
                files.extend(md5_map[digest])
            clusters.append(sorted(files))
    return clusters


def build_delete_plan(folder: str, md5_map: dict, clusters: list[list[str]]) -> dict:
    """각 클러스터/md5그룹마다 해상도가 가장 큰(동률이면 이름이 빠른) 파일만 남긴다."""
    delete_files = []
    reasons = {}
    clustered_digests = set()

    size_cache = {}

    def area(fname):
        if fname not in size_cache:
            with Image.open(os.path.join(folder, fname)) as im:
                size_cache[fname] = im.size[0] * im.size[1]
        return size_cache[fname]

    for cluster_files in clusters:
        keeper = sorted(cluster_files, key=lambda f: (-area(f), f))[0]
        for f in cluster_files:
            if f != keeper:
                delete_files.append(f)
                reasons[f] = f"perceptual-dup-of:{keeper}"

    # 이미 클러스터에 포함된 파일은 건너뛰고, 남은 md5 완전중복 그룹만 별도 처리
    already_handled = set(delete_files) | {f for c in clusters for f in c}
    for digest, flist in md5_map.items():
        if len(flist) <= 1 or any(f in already_handled for f in flist):
            continue
        flist_sorted = sorted(flist)
        keeper = flist_sorted[0]
        for f in flist_sorted[1:]:
            delete_files.append(f)
            reasons[f] = f"exact-md5-dup-of:{keeper}"

    return {"delete_files": sorted(delete_files), "reasons": reasons}


def move_to_trash(folder: str, files: list[str], batch_size: int = 50):
    """AppleScript(Finder)로 휴지통 이동. rm과 달리 되돌릴 수 있다."""
    paths = [os.path.join(folder, f) for f in files]
    for i in range(0, len(paths), batch_size):
        batch = paths[i : i + batch_size]
        posix_list = ", ".join('"' + p + '"' for p in batch)
        script = f"""
        set fileList to {{{posix_list}}}
        tell application "Finder"
            repeat with f in fileList
                set thisPath to (contents of f) as text
                move (POSIX file thisPath as alias) to trash
            end repeat
        end tell
        """
        for attempt in range(3):
            result = subprocess.run(["osascript", "-e", script], capture_output=True, text=True)
            if result.returncode == 0:
                break
            time.sleep(2)
        else:
            print(f"FAILED batch starting at {i}: {result.stderr.strip()}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", help="GIF 파일들이 있는 폴더 경로")
    parser.add_argument("--apply", action="store_true", help="실제로 휴지통으로 이동 (기본은 dry-run)")
    parser.add_argument("--plan-out", default="dedup_plan.json", help="삭제 계획을 저장할 JSON 경로")
    args = parser.parse_args()

    print("1/3 MD5로 완전 동일 파일 탐색 중...")
    md5_map = scan_md5(args.folder)
    exact_dupe_groups = {k: v for k, v in md5_map.items() if len(v) > 1}
    print(f"  고유 콘텐츠 그룹: {len(md5_map)} / 완전중복 그룹: {len(exact_dupe_groups)}")

    print("2/3 perceptual hash로 재업로드된 동일 영상 탐색 중...")
    results = compute_phash(args.folder, md5_map)
    clusters = cluster_by_similarity(md5_map, results)
    print(f"  perceptual 클러스터: {len(clusters)}개")

    plan = build_delete_plan(args.folder, md5_map, clusters)
    print(f"3/3 삭제 대상: {len(plan['delete_files'])}개")

    with open(args.plan_out, "w") as f:
        json.dump(plan, f, ensure_ascii=False, indent=1)
    print(f"계획 저장: {args.plan_out}")

    if args.apply:
        print("휴지통으로 이동 중...")
        move_to_trash(args.folder, plan["delete_files"])
        print("완료")
    else:
        print("dry-run 모드입니다. 실제 삭제하려면 --apply 옵션을 추가하세요.")


if __name__ == "__main__":
    main()
