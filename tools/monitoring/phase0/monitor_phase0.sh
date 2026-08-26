#!/usr/bin/env bash

# AFMC Phase-0 live benchmark monitor
# READ-ONLY: does not modify benchmark state.

set -u

ROOT="/home/royja/afmc-clinical-fm"
REFRESH="${REFRESH:-3}"
EXPECTED_CELLS=2730
EXPECTED_SHARDS=25

cd "$ROOT" || exit 1

SHA="$(git rev-parse HEAD 2>/dev/null)"
OUT="outputs/phase0_full_cuda_${SHA}"

RESET=$'\033[0m'
BOLD=$'\033[1m'
DIM=$'\033[2m'
GREEN=$'\033[32m'
YELLOW=$'\033[33m'
CYAN=$'\033[36m'
RED=$'\033[31m'

trap 'printf "\033[?25h\n"; exit' INT TERM EXIT
printf '\033[?25l'

fmt_time() {
    local s="${1:-0}"
    (( s < 0 )) && s=0

    local h=$((s / 3600))
    local m=$(((s % 3600) / 60))
    local sec=$((s % 60))

    if (( h > 0 )); then
        printf "%02dh %02dm %02ds" "$h" "$m" "$sec"
    else
        printf "%02dm %02ds" "$m" "$sec"
    fi
}

bar() {
    local current="$1"
    local total="$2"
    local width="${3:-48}"

    local filled=0
    if (( total > 0 )); then
        filled=$(( current * width / total ))
    fi

    (( filled > width )) && filled="$width"

    local empty=$(( width - filled ))

    printf "${GREEN}"
    printf "%${filled}s" "" | tr ' ' '█'
    printf "${DIM}"
    printf "%${empty}s" "" | tr ' ' '░'
    printf "${RESET}"
}

get_run_start() {
    if [[ -f "$OUT/run_record.json" ]]; then
        ./.venv/bin/python - "$OUT/run_record.json" 2>/dev/null <<'PY'
import json
import sys
from datetime import datetime

try:
    with open(sys.argv[1], encoding="utf-8") as f:
        d = json.load(f)

    x = d.get("original_started_at")
    if x:
        dt = datetime.fromisoformat(x.replace("Z", "+00:00"))
        print(int(dt.timestamp()))
except Exception:
    pass
PY
    fi
}

MONITOR_START="$(date +%s)"
RUN_START="$(get_run_start)"
[[ -z "${RUN_START:-}" ]] && RUN_START="$MONITOR_START"

previous_cells=0
previous_time="$(date +%s)"
first_sample=1

while true; do
    now="$(date +%s)"

    cells="$(
        find "$OUT/shards" \
            -type f \
            -path '*/cells/*.json' \
            2>/dev/null |
        wc -l
    )"

    completed_shards="$(
        find "$OUT/shards" \
            -type f \
            -name COMPLETE \
            2>/dev/null |
        wc -l
    )"

    observed_shards="$(
        find "$OUT/shards" \
            -mindepth 1 \
            -maxdepth 1 \
            -type d \
            2>/dev/null |
        wc -l
    )"

    elapsed=$(( now - RUN_START ))
    (( elapsed < 1 )) && elapsed=1

    remaining=$(( EXPECTED_CELLS - cells ))
    (( remaining < 0 )) && remaining=0

    percent="$(
        awk -v c="$cells" -v t="$EXPECTED_CELLS" \
            'BEGIN { printf "%.2f", (t > 0 ? 100*c/t : 0) }'
    )"

    avg_rate="$(
        awk -v c="$cells" -v e="$elapsed" \
            'BEGIN { printf "%.2f", (e > 0 ? c*60/e : 0) }'
    )"

    interval=$(( now - previous_time ))

    if (( first_sample == 1 || interval <= 0 )); then
        recent_rate="--"
        first_sample=0
    else
        delta=$(( cells - previous_cells ))
        recent_rate="$(
            awk -v d="$delta" -v s="$interval" \
                'BEGIN { printf "%.2f", (s > 0 ? d*60/s : 0) }'
        )"
    fi

    eta="$(
        awk -v c="$cells" -v t="$EXPECTED_CELLS" -v r="$avg_rate" '
        BEGIN {
            if (r > 0 && t > c)
                printf "%.0f", ((t-c)/r)*60;
            else
                print 0;
        }'
    )"

    benchmark_processes="$(
        pgrep -af 'afmc_fm.cli benchmark|multiprocessing.spawn|spawn_main' \
            2>/dev/null |
        grep -v monitor_phase0 |
        wc -l
    )"

    printf '\033[H\033[2J'

    printf "${BOLD}${CYAN}"
    printf "┌────────────────────────────────────────────────────────────────────────────────────┐\n"
    printf "│                         AFMC PHASE-0 CUDA BENCHMARK                                │\n"
    printf "└────────────────────────────────────────────────────────────────────────────────────┘\n"
    printf "${RESET}"

    printf "\n${BOLD}Execution${RESET}\n"
    printf "  SHA       ${DIM}%s${RESET}\n" "$SHA"
    printf "  Output    ${DIM}%s${RESET}\n" "$OUT"
    printf "  Time      %s\n" "$(date '+%Y-%m-%d %H:%M:%S %Z')"
    printf "  Elapsed   ${BOLD}%s${RESET}\n" "$(fmt_time "$elapsed")"

    printf "\n${BOLD}Overall progress${RESET}\n\n"

    printf "  "
    bar "$cells" "$EXPECTED_CELLS" 52
    printf "  ${BOLD}%6s%%${RESET}\n" "$percent"

    printf "\n"
    printf "  Cells             ${GREEN}${BOLD}%4d${RESET} / %-4d" \
        "$cells" "$EXPECTED_CELLS"
    printf "       Remaining   ${BOLD}%4d${RESET}\n" "$remaining"

    printf "  Completed shards  ${CYAN}${BOLD}%4d${RESET} / %-4d" \
        "$completed_shards" "$EXPECTED_SHARDS"
    printf "       Observed    ${BOLD}%4d${RESET}\n" "$observed_shards"

    printf "  Processes         ${BOLD}%4d${RESET}\n" "$benchmark_processes"

    printf "\n${BOLD}Throughput${RESET}\n"
    printf "  Average        ${BOLD}%8s${RESET} cells/min\n" "$avg_rate"
    printf "  Recent         ${BOLD}%8s${RESET} cells/min\n" "$recent_rate"

    if (( cells >= EXPECTED_CELLS )); then
        printf "  ETA            ${GREEN}${BOLD}COMPLETE${RESET}\n"
    elif [[ "$avg_rate" != "0.00" ]]; then
        printf "  Estimated ETA  ${YELLOW}${BOLD}%s${RESET}\n" "$(fmt_time "$eta")"
    else
        printf "  Estimated ETA  ${DIM}calculating...${RESET}\n"
    fi

    printf "\n${BOLD}GPU${RESET}\n"

    if command -v nvidia-smi >/dev/null 2>&1; then
        gpu="$(
            nvidia-smi \
                --query-gpu=name,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw \
                --format=csv,noheader,nounits \
                2>/dev/null |
            head -n 1
        )"

        if [[ -n "$gpu" ]]; then
            IFS=',' read -r gpu_name gpu_util mem_used mem_total temp power <<< "$gpu"

            printf "  %-11s ${BOLD}%s${RESET}\n" \
                "Device" "$(echo "$gpu_name" | xargs)"

            printf "  %-11s ${BOLD}%s%%${RESET}" \
                "Utilization" "$(echo "$gpu_util" | xargs)"

            printf "       VRAM ${BOLD}%s / %s MiB${RESET}" \
                "$(echo "$mem_used" | xargs)" \
                "$(echo "$mem_total" | xargs)"

            printf "       Temp ${BOLD}%s°C${RESET}" \
                "$(echo "$temp" | xargs)"

            printf "       Power ${BOLD}%s W${RESET}\n" \
                "$(echo "$power" | xargs)"
        else
            printf "  ${RED}No GPU telemetry returned.${RESET}\n"
        fi
    else
        printf "  ${RED}nvidia-smi unavailable.${RESET}\n"
    fi

    # ------------------------------------------------------------------
    # ACTIVE / INCOMPLETE SHARDS
    # ------------------------------------------------------------------

    printf "\n${BOLD}${YELLOW}ONGOING / INCOMPLETE SHARDS${RESET}\n"
    printf "  ${DIM}cells   status      shard${RESET}\n"
    printf "  ${DIM}─────   ─────────   ───────────────────────────────────────────────────────────${RESET}\n"

    active_count=0

    if [[ -d "$OUT/shards" ]]; then
        while IFS=$'\t' read -r count shard; do
            [[ -n "${shard:-}" ]] || continue
            active_count=$(( active_count + 1 ))

            printf "  ${YELLOW}${BOLD}%5s${RESET}   ${YELLOW}RUNNING${RESET}     %s\n" \
                "$count" "$shard"

        done < <(
            for d in "$OUT"/shards/*; do
                [[ -d "$d" ]] || continue
                [[ -f "$d/COMPLETE" ]] && continue

                count="$(
                    find "$d/cells" \
                        -maxdepth 1 \
                        -type f \
                        -name '*.json' \
                        2>/dev/null |
                    wc -l
                )"

                printf "%s\t%s\n" \
                    "$count" "$(basename "$d")"
            done |
            sort -t$'\t' -k1,1nr
        )
    fi

    if (( active_count == 0 )); then
        if (( completed_shards >= EXPECTED_SHARDS )); then
            printf "  ${GREEN}${BOLD}No incomplete shards — all shards complete.${RESET}\n"
        else
            printf "  ${DIM}Waiting for next shard to begin...${RESET}\n"
        fi
    fi

    # ------------------------------------------------------------------
    # COMPLETED SHARDS
    # ------------------------------------------------------------------

    printf "\n${BOLD}${GREEN}COMPLETED SHARDS${RESET}\n"
    printf "  ${DIM}cells   status      shard${RESET}\n"
    printf "  ${DIM}─────   ─────────   ───────────────────────────────────────────────────────────${RESET}\n"

    complete_count=0

    if [[ -d "$OUT/shards" ]]; then
        while IFS=$'\t' read -r count shard; do
            [[ -n "${shard:-}" ]] || continue
            complete_count=$(( complete_count + 1 ))

            printf "  ${GREEN}${BOLD}%5s${RESET}   ${GREEN}COMPLETE${RESET}    %s\n" \
                "$count" "$shard"

        done < <(
            for d in "$OUT"/shards/*; do
                [[ -d "$d" ]] || continue
                [[ -f "$d/COMPLETE" ]] || continue

                count="$(
                    find "$d/cells" \
                        -maxdepth 1 \
                        -type f \
                        -name '*.json' \
                        2>/dev/null |
                    wc -l
                )"

                printf "%s\t%s\n" \
                    "$count" "$(basename "$d")"
            done |
            sort -t$'\t' -k2,2
        )
    fi

    if (( complete_count == 0 )); then
        printf "  ${DIM}No shards completed yet.${RESET}\n"
    fi

    printf "\n${DIM}"
    printf "Ongoing: %d   Complete: %d/%d   Refresh: %ss   Ctrl+C exits monitor only\n" \
        "$active_count" \
        "$completed_shards" \
        "$EXPECTED_SHARDS" \
        "$REFRESH"
    printf "${RESET}"

    previous_cells="$cells"
    previous_time="$now"

    sleep "$REFRESH"
done
