#!/usr/bin/env bash
set -u

CODE_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
OUT_DIR="${CODE_DIR}/../output"
PROJECT_DIR="${CODE_DIR}/../../_lib/cdp_julia"

mkdir -p "${OUT_DIR}"

{
  echo "date=$(date '+%Y-%m-%d %H:%M:%S')"
  echo "project_dir=${PROJECT_DIR}"
  if command -v julia >/dev/null 2>&1; then
    echo "julia_found=true"
    julia --version 2>/dev/null || echo "julia_version_check_failed=true"
  else
    echo "julia_found=false"
  fi
} > "${OUT_DIR}/environment_report.txt"

{
  echo "package,status"
  if command -v julia >/dev/null 2>&1; then
    julia --project="${PROJECT_DIR}" -e 'using MAT, JLD2, CSV, DataFrames; println("MAT,ok\nJLD2,ok\nCSV,ok\nDataFrames,ok")' 2>/dev/null \
      || echo "MAT,missing_or_error"
  else
    echo "MAT,julia_not_found"
    echo "JLD2,julia_not_found"
    echo "CSV,julia_not_found"
    echo "DataFrames,julia_not_found"
  fi
} > "${OUT_DIR}/julia_packages_status.txt"
