#!/usr/bin/env python3
"""KAMP 2026 공개 전 준비 및 공개 후 검증 도구. 가상 대회 데이터는 생성하지 않는다."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import shutil
import subprocess
import sys
import zipfile
from datetime import datetime
from importlib import metadata
from pathlib import Path
from typing import Any


REQUIRED = {
    "numpy": "1.26.4",
    "pandas": "2.2.2",
    "scipy": "1.13.1",
    "scikit-learn": "1.5.1",
    "pyarrow": "16.1.0",
    "openpyxl": "3.1.5",
    "jupyterlab": "4.2.5",
    "pytest": "7.4.4",
}
TABULAR_EXTENSIONS = {".csv", ".parquet", ".xlsx", ".jsonl"}


class PrepError(Exception):
    """User-correctable setup or data error."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def save_json(path: Path, content: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise PrepError(f"기존 보고서를 덮어쓰지 않습니다: {path}")
    with path.open("x", encoding="utf-8") as stream:
        json.dump(content, stream, ensure_ascii=False, indent=2, default=str)
        stream.write("\n")


def package_versions() -> dict[str, str | None]:
    result: dict[str, str | None] = {}
    for name in REQUIRED:
        try:
            result[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            result[name] = None
    return result


def installed_packages() -> list[str]:
    return sorted(f"{distribution.metadata['Name']}=={distribution.version}"
                  for distribution in metadata.distributions() if distribution.metadata.get("Name"))


def environment(args: argparse.Namespace) -> None:
    versions = package_versions()
    mismatch = {name: {"expected": wanted, "actual": versions[name]}
                for name, wanted in REQUIRED.items() if versions[name] != wanted}
    disk = shutil.disk_usage(Path.cwd())
    memory_gib = None
    cpu_brand = platform.processor() or None
    if sys.platform == "darwin":
        try:
            memory_gib = round(int(subprocess.check_output(["sysctl", "-n", "hw.memsize"], text=True).strip()) / 1024**3, 2)
            cpu_brand = subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"], text=True).strip()
        except (OSError, subprocess.CalledProcessError, ValueError):
            pass
    result = {
        "checked_at": datetime.now().astimezone().isoformat(),
        "python": sys.version.split()[0],
        "executable": sys.executable,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "cpu_brand": cpu_brand,
        "logical_cpus": os.cpu_count(),
        "memory_gib": memory_gib,
        "disk_free_gib": round(disk.free / 1024**3, 2),
        "packages": versions,
        "installed_packages": installed_packages(),
        "package_mismatches": mismatch,
        "status": "PASS" if sys.version_info[:2] == (3, 12) and not mismatch else "CHECK",
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if args.report:
        save_json(args.report, result)
    if result["status"] != "PASS":
        raise PrepError("Python 3.12 또는 고정 패키지 버전을 확인하세요.")


def archive_info(path: Path) -> dict[str, Any]:
    entries = []
    with zipfile.ZipFile(path) as archive:
        for item in archive.infolist():
            name = item.filename.replace("\\", "/")
            parts = Path(name).parts
            unsafe = name.startswith("/") or ".." in parts or (bool(parts) and ":" in parts[0])
            entries.append({"name": name, "bytes": item.file_size, "unsafe_path": unsafe})
    return {"entries": entries, "unsafe_paths": [x["name"] for x in entries if x["unsafe_path"]]}


def inventory(args: argparse.Namespace) -> None:
    root = args.directory.resolve()
    if not root.is_dir():
        raise PrepError(f"수신 폴더가 없습니다: {root}")
    files = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.name == ".gitkeep":
            continue
        if path.is_symlink():
            raise PrepError(f"심볼릭 링크는 검토 후 별도 처리하세요: {path}")
        entry: dict[str, Any] = {
            "path": str(path.relative_to(root)),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        if path.suffix.lower() == ".zip":
            entry["zip"] = archive_info(path)
        files.append(entry)
    result = {"directory": str(root), "file_count": len(files), "files": files}
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if args.report:
        save_json(args.report, result)


def csv_format(path: Path) -> tuple[str, str, list[str]]:
    sample_bytes = path.open("rb").read(65536)
    if not sample_bytes:
        raise PrepError(f"빈 CSV 파일입니다: {path}")
    for encoding in ("utf-8-sig", "cp949", "euc-kr"):
        try:
            sample = sample_bytes.decode(encoding)
            break
        except UnicodeDecodeError:
            pass
    else:
        raise PrepError("CSV 인코딩을 자동 판별하지 못했습니다. 원본 설명서를 확인하세요.")
    try:
        delimiter = csv.Sniffer().sniff(sample, delimiters=",\t;|").delimiter
    except csv.Error:
        delimiter = ","
    header = next(csv.reader(sample.splitlines(), delimiter=delimiter), [])
    if not header:
        raise PrepError("CSV 헤더를 찾지 못했습니다.")
    return encoding, delimiter, header


def sample_table(path: Path, rows: int = 1000, sheet: str | None = None):
    """Read only the first rows; return (DataFrame, source metadata)."""
    import pandas as pd

    if rows < 1:
        raise PrepError("샘플 행 수는 1 이상이어야 합니다.")
    if not path.is_file():
        raise PrepError(f"파일이 없습니다: {path}")
    suffix = path.suffix.lower()
    if suffix == ".csv":
        encoding, delimiter, header = csv_format(path)
        frame = pd.read_csv(path, encoding=encoding, sep=delimiter, nrows=rows)
        return frame, {"encoding": encoding, "delimiter": delimiter,
                       "header": header, "duplicate_headers": sorted({x for x in header if header.count(x) > 1})}
    if suffix == ".parquet":
        import pyarrow.parquet as pq

        parquet = pq.ParquetFile(path)
        batch = next(parquet.iter_batches(batch_size=rows), None)
        frame = batch.to_pandas() if batch is not None else pd.DataFrame(columns=parquet.schema_arrow.names)
        return frame, {"total_rows": parquet.metadata.num_rows,
                       "arrow_schema": str(parquet.schema_arrow),
                       "duplicate_headers": sorted({x for x in parquet.schema_arrow.names
                                                    if parquet.schema_arrow.names.count(x) > 1})}
    if suffix == ".xlsx":
        book = pd.ExcelFile(path)
        if len(book.sheet_names) > 1 and sheet is None:
            raise PrepError(f"시트가 여러 개입니다. --sheet를 지정하세요: {book.sheet_names}")
        selected = sheet or book.sheet_names[0]
        if selected not in book.sheet_names:
            raise PrepError(f"없는 시트입니다: {selected}; 실제 시트: {book.sheet_names}")
        frame = pd.read_excel(book, sheet_name=selected, nrows=rows)
        return frame, {"sheet_names": book.sheet_names, "selected_sheet": selected}
    if suffix == ".jsonl":
        frame = pd.read_json(path, lines=True, nrows=rows)
        return frame, {}
    raise PrepError(f"스키마 샘플링 미지원 형식: {suffix}. 지원: {sorted(TABULAR_EXTENSIONS)}")


def inspect(args: argparse.Namespace) -> None:
    path = args.file.resolve()
    frame, info = sample_table(path, args.rows, args.sheet)
    if info.get("duplicate_headers"):
        raise PrepError(f"중복 열 이름을 먼저 검토하세요: {info['duplicate_headers']}")
    columns = []
    for name in frame.columns:
        series = frame[name]
        columns.append({"name": str(name), "sample_dtype": str(series.dtype),
                        "sample_nulls": int(series.isna().sum()),
                        "sample_distinct": int(series.nunique(dropna=True))})
    result = {"file": str(path), "bytes": path.stat().st_size,
              "sha256": sha256_file(path), "sample_rows": len(frame),
              "total_columns": len(frame.columns), "source_info": info, "columns": columns,
              "warning": "타입/결측/고유값은 앞부분 샘플 기준입니다. 전체 데이터와 공식 설명서를 대조하세요."}
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if args.report:
        save_json(args.report, result)


def load_config(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise PrepError(f"설정 파일이 없습니다: {path}. 공개 후 템플릿을 복사·확정하세요.")
    try:
        with path.open(encoding="utf-8") as stream:
            value = json.load(stream)
    except (OSError, json.JSONDecodeError) as error:
        raise PrepError(f"설정 파일을 읽을 수 없습니다: {error}") from error
    if not isinstance(value, dict):
        raise PrepError("설정 파일의 최상위 구조는 JSON 객체여야 합니다.")
    return value


def required(value: Any, label: str) -> Any:
    if value is None or value == "" or value == []:
        raise PrepError(f"공식 과제를 확인한 뒤 {label}을(를) 설정하세요. 임의 값으로 진행하지 않습니다.")
    return value


def read_csv_as_text(path: Path):
    import pandas as pd

    encoding, delimiter, header = csv_format(path)
    if len(header) != len(set(header)):
        raise PrepError(f"중복 CSV 헤더가 있습니다: {path}")
    frame = pd.read_csv(path, encoding=encoding, sep=delimiter, dtype=str, keep_default_na=False)
    return frame, header


def check_output(args: argparse.Namespace) -> None:
    spec = load_config(args.config).get("submission", {})
    candidate = args.candidate.resolve()
    if candidate.suffix.lower() != ".csv":
        raise PrepError("현재 검사기는 CSV 제출물만 지원합니다. 공식 형식을 먼저 확인하세요.")
    if not candidate.is_file():
        raise PrepError(f"결과 파일이 없습니다: {candidate}")
    predictions = required(spec.get("prediction_columns"), "submission.prediction_columns")
    if not isinstance(predictions, list) or not all(isinstance(x, str) for x in predictions):
        raise PrepError("prediction_columns는 열 이름 문자열 목록이어야 합니다.")
    reference = spec.get("reference_file")
    if reference:
        reference_path = Path(reference)
        reference_frame, expected = read_csv_as_text(reference_path)
        expected_rows = len(reference_frame)
    else:
        expected = required(spec.get("expected_columns"), "submission.expected_columns")
        expected_rows = required(spec.get("expected_rows"), "submission.expected_rows")
        reference_frame = None
    if not isinstance(expected, list) or not all(isinstance(x, str) for x in expected):
        raise PrepError("expected_columns는 열 이름 문자열 목록이어야 합니다.")
    if not isinstance(expected_rows, int) or expected_rows < 1:
        raise PrepError("expected_rows는 양의 정수여야 합니다.")
    frame, actual = read_csv_as_text(candidate)
    errors = []
    if actual != expected:
        errors.append(f"열 이름/순서 불일치: 기대 {expected}, 실제 {actual}")
    if len(frame) != expected_rows:
        errors.append(f"행 수 불일치: 기대 {expected_rows}, 실제 {len(frame)}")
    id_column = spec.get("id_column")
    if id_column:
        if id_column not in frame:
            errors.append(f"ID 열 없음: {id_column}")
        else:
            ids = frame[id_column].astype(str)
            if ids.duplicated().any():
                errors.append(f"ID 중복 {int(ids.duplicated().sum())}건")
            if ids.str.strip().eq("").any():
                errors.append("빈 ID 발견")
            if reference_frame is not None:
                if id_column not in reference_frame:
                    errors.append(f"참조 파일에 ID 열 없음: {id_column}")
                elif ids.tolist() != reference_frame[id_column].astype(str).tolist():
                    errors.append("공식 참조 파일과 ID 또는 행 순서 불일치")
    allowed = spec.get("allowed_values") or {}
    numeric = spec.get("numeric_columns") or []
    for column in predictions:
        if column not in frame:
            errors.append(f"예측 열 없음: {column}")
            continue
        values = frame[column].astype(str)
        if values.str.strip().eq("").any():
            errors.append(f"예측값 공백: {column}")
        if column in numeric:
            import pandas as pd

            numbers = pd.to_numeric(values, errors="coerce")
            if numbers.isna().any() or not numbers.map(math.isfinite).all():
                errors.append(f"숫자 아닌 값/NaN/무한대: {column}")
        if column in allowed:
            bad = ~values.isin([str(x) for x in allowed[column]])
            if bad.any():
                errors.append(f"허용되지 않은 레이블 {int(bad.sum())}건: {column}")
    result = {"candidate": str(candidate), "sha256": sha256_file(candidate),
              "rows": len(frame), "columns": actual, "status": "PASS" if not errors else "FAIL",
              "errors": errors}
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if errors:
        raise PrepError("결과 파일 검사를 통과하지 못했습니다.")


def split_indices(frame, config: dict[str, Any], target: str, task_type: str):
    import numpy as np
    import pandas as pd
    from sklearn.model_selection import GroupShuffleSplit, train_test_split

    spec = config.get("split", {})
    strategy = required(spec.get("strategy"), "split.strategy")
    fraction = spec.get("test_size")
    seed = spec.get("seed")
    if not isinstance(fraction, (int, float)) or not 0 < fraction < 1:
        raise PrepError("split.test_size는 0과 1 사이여야 합니다.")
    if not isinstance(seed, int):
        raise PrepError("split.seed는 정수여야 합니다.")
    index = np.arange(len(frame))
    if strategy in ("random", "stratified"):
        if strategy == "stratified" and task_type != "classification":
            raise PrepError("stratified는 분류 과제에만 사용합니다.")
        return train_test_split(index, test_size=fraction, random_state=seed,
                                stratify=frame[target] if strategy == "stratified" else None)
    if strategy == "group":
        column = required(spec.get("group_column"), "split.group_column")
        if column not in frame or frame[column].isna().any():
            raise PrepError("그룹 열이 없거나 결측이 있습니다.")
        splitter = GroupShuffleSplit(n_splits=1, test_size=fraction, random_state=seed)
        return next(splitter.split(index, groups=frame[column]))
    if strategy == "time":
        column = required(spec.get("time_column"), "split.time_column")
        if column not in frame:
            raise PrepError(f"시간 열이 없습니다: {column}")
        times = pd.to_datetime(frame[column], errors="raise", utc=True)
        if times.isna().any():
            raise PrepError("시간 열에 결측이 있습니다.")
        order = np.argsort(times.to_numpy(), kind="stable")
        cutoff = int(len(frame) * (1 - fraction))
        boundary_time = times.iloc[order[cutoff]]
        while cutoff > 0 and times.iloc[order[cutoff - 1]] == boundary_time:
            cutoff -= 1
        if cutoff == 0 or cutoff == len(frame):
            raise PrepError("같은 시각이 분할 경계에 걸려 시간 분할이 불가능합니다.")
        return order[:cutoff], order[cutoff:]
    raise PrepError(f"지원하지 않는 분할: {strategy}. 공식 분할이 있다면 별도 구현하세요.")


def baseline(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    source = Path(required(config.get("source_file"), "source_file"))
    target = required(config.get("target_column"), "target_column")
    task_type = required(config.get("task_type"), "task_type")
    metric = required(config.get("metric"), "metric")
    required(config.get("split", {}).get("strategy"), "split.strategy")
    if task_type not in ("classification", "regression"):
        raise PrepError("이 CPU 기준선은 표 형태의 분류/회귀 지도학습만 지원합니다.")
    supported = {"classification": {"accuracy", "balanced_accuracy", "f1_macro", "roc_auc", "log_loss"},
                 "regression": {"rmse", "mae", "r2"}}
    if metric not in supported[task_type]:
        raise PrepError(f"이 기준선의 지원 지표가 아닙니다: {metric}. 공식 지표에 맞게 확장하세요.")
    if source.suffix.lower() not in TABULAR_EXTENSIONS:
        raise PrepError("이 기준선은 표 형태 파일만 지원합니다.")
    if not source.is_file():
        raise PrepError(f"실제 학습 파일이 없습니다: {source}")
    if source.stat().st_size > 512 * 1024**2 and not args.allow_large:
        raise PrepError("원본이 512MiB를 넘습니다. 메모리/형식을 검토한 뒤 --allow-large로 명시하세요.")
    import joblib
    import numpy as np
    import pandas as pd
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression, Ridge
    from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score,
                                 log_loss, mean_absolute_error, mean_squared_error,
                                 r2_score, roc_auc_score)
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    if source.suffix.lower() == ".csv":
        encoding, delimiter, _ = csv_format(source)
        frame = pd.read_csv(source, encoding=encoding, sep=delimiter)
    elif source.suffix.lower() == ".parquet":
        frame = pd.read_parquet(source)
    elif source.suffix.lower() == ".xlsx":
        frame = pd.read_excel(source)
    else:
        frame = pd.read_json(source, lines=True)
    if target not in frame:
        raise PrepError(f"타깃 열이 없습니다: {target}")
    if frame[target].isna().any() or len(frame) < 10:
        raise PrepError("타깃 결측이 있거나 학습 행이 10개 미만입니다.")
    train_idx, valid_idx = split_indices(frame, config, target, task_type)
    excluded = set(config.get("exclude_columns") or []) | {target}
    split_spec = config["split"]
    for key in ("group_column", "time_column"):
        if split_spec.get(key):
            excluded.add(split_spec[key])
    if config.get("submission", {}).get("id_column"):
        excluded.add(config["submission"]["id_column"])
    missing_excluded = excluded - set(frame.columns)
    if missing_excluded:
        raise PrepError(f"지정한 열이 학습 데이터에 없습니다: {sorted(missing_excluded)}")
    x = frame.drop(columns=list(excluded))
    if x.empty or x.shape[1] == 0:
        raise PrepError("사용 가능한 입력 특징이 없습니다.")
    y = frame[target]
    numeric_cols = x.select_dtypes(include="number").columns.tolist()
    categorical_cols = [c for c in x.columns if c not in numeric_cols]
    x = x.copy()
    for column in categorical_cols:
        x[column] = x[column].astype("object").where(x[column].notna(), np.nan)
    transformers = []
    if numeric_cols:
        transformers.append(("numeric", make_pipeline(SimpleImputer(strategy="median"),
                                                      StandardScaler(with_mean=False)), numeric_cols))
    if categorical_cols:
        transformers.append(("categorical", make_pipeline(SimpleImputer(strategy="most_frequent"),
                                                          OneHotEncoder(handle_unknown="ignore")), categorical_cols))
    prep = ColumnTransformer(transformers)
    estimator = LogisticRegression(max_iter=500, random_state=split_spec["seed"]) if task_type == "classification" else Ridge()
    model = make_pipeline(prep, estimator)
    model.fit(x.iloc[train_idx], y.iloc[train_idx])
    true = y.iloc[valid_idx]
    predicted = model.predict(x.iloc[valid_idx])
    if metric == "accuracy":
        score = accuracy_score(true, predicted)
    elif metric == "balanced_accuracy":
        score = balanced_accuracy_score(true, predicted)
    elif metric == "f1_macro":
        score = f1_score(true, predicted, average="macro")
    elif metric == "roc_auc":
        if len(model.classes_) != 2:
            raise PrepError("roc_auc 기준선은 이진 분류만 지원합니다.")
        score = roc_auc_score(true, model.predict_proba(x.iloc[valid_idx])[:, 1])
    elif metric == "log_loss":
        score = log_loss(true, model.predict_proba(x.iloc[valid_idx]), labels=model.classes_)
    elif metric == "rmse":
        score = mean_squared_error(true, predicted) ** 0.5
    elif metric == "mae":
        score = mean_absolute_error(true, predicted)
    else:
        score = r2_score(true, predicted)
    split_digest = hashlib.sha256(np.asarray(train_idx, dtype="<i8").tobytes() +
                                  b"/" + np.asarray(valid_idx, dtype="<i8").tobytes()).hexdigest()
    run_dir = Path("reports") / f"baseline-{datetime.now().strftime('%Y%m%d-%H%M%S-%f')}"
    run_dir.mkdir(parents=True, exist_ok=False)
    result = {"source_file": str(source), "source_sha256": sha256_file(source),
              "config_sha256": sha256_file(args.config), "task_type": task_type,
              "metric": metric, "score": float(score), "split": split_spec,
              "split_sha256": split_digest, "train_rows": len(train_idx),
              "validation_rows": len(valid_idx), "numeric_columns": numeric_cols,
              "categorical_columns": categorical_cols, "packages": package_versions(),
              "installed_packages": installed_packages(),
              "python": sys.version.split()[0],
              "caution": "일반 CPU 기준선입니다. 공식 규칙·누수·분할 적합성을 별도로 확인하세요."}
    save_json(run_dir / "result.json", result)
    joblib.dump(model, run_dir / "model.joblib")
    print(json.dumps({"run_dir": str(run_dir), **result}, ensure_ascii=False, indent=2))


def compare_runs(args: argparse.Namespace) -> None:
    if args.atol < 0 or not math.isfinite(args.atol):
        raise PrepError("--atol은 0 이상의 유한한 수여야 합니다.")
    previous = load_config(args.previous)
    current = load_config(args.current)
    keys = ("source_sha256", "config_sha256", "task_type", "metric", "split_sha256",
            "train_rows", "validation_rows", "numeric_columns", "categorical_columns", "packages",
            "installed_packages", "python")
    differences = {key: {"previous": previous.get(key), "current": current.get(key)}
                   for key in keys if previous.get(key) != current.get(key)}
    try:
        old_score = float(previous["score"])
        new_score = float(current["score"])
    except (KeyError, TypeError, ValueError) as error:
        raise PrepError("두 파일 모두 기준선 result.json이어야 합니다.") from error
    if not math.isfinite(old_score) or not math.isfinite(new_score):
        raise PrepError("비유한 점수는 재현 비교할 수 없습니다.")
    difference = abs(old_score - new_score)
    if difference > args.atol:
        differences["score"] = {"previous": old_score, "current": new_score,
                                "absolute_difference": difference, "allowed": args.atol}
    result = {"status": "PASS" if not differences else "FAIL",
              "previous": str(args.previous), "current": str(args.current),
              "score_absolute_difference": difference, "differences": differences}
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if differences:
        raise PrepError("재현 점검에 차이가 있습니다. 원인 확인 전 최종본으로 사용하지 마세요.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    env = commands.add_parser("env", help="Python/CPU 환경·패키지 버전 확인")
    env.add_argument("--report", type=Path)
    env.set_defaults(action=environment)
    listing = commands.add_parser("inventory", help="받은 원본 파일의 SHA-256과 ZIP 목록")
    listing.add_argument("directory", type=Path)
    listing.add_argument("--report", type=Path)
    listing.set_defaults(action=inventory)
    profiling = commands.add_parser("inspect", help="실제 입력 파일 앞부분의 열·타입·결측 확인")
    profiling.add_argument("file", type=Path)
    profiling.add_argument("--rows", type=int, default=1000)
    profiling.add_argument("--sheet")
    profiling.add_argument("--report", type=Path)
    profiling.set_defaults(action=inspect)
    output = commands.add_parser("check-output", help="설정과 공식 참조 파일에 따른 CSV 제출물 검사")
    output.add_argument("candidate", type=Path)
    output.add_argument("--config", type=Path, required=True)
    output.set_defaults(action=check_output)
    training = commands.add_parser("baseline", help="공개 후에만 실행하는 표 데이터 CPU 기준선")
    training.add_argument("--config", type=Path, required=True)
    training.add_argument("--allow-large", action="store_true")
    training.set_defaults(action=baseline)
    comparison = commands.add_parser("compare-runs", help="10/3 기준선 재현 기록 비교")
    comparison.add_argument("previous", type=Path)
    comparison.add_argument("current", type=Path)
    comparison.add_argument("--atol", type=float, default=1e-8)
    comparison.set_defaults(action=compare_runs)
    args = parser.parse_args()
    try:
        args.action(args)
    except (PrepError, OSError, ValueError, zipfile.BadZipFile) as error:
        parser.exit(2, f"오류: {error}\n")


if __name__ == "__main__":
    main()
