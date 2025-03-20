#!/bin/bash

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "${SCRIPT_DIR}/../src"
DATADIR="../datasets/Psychological_Datasets/"

# Initialize arrays and counters
accuracy_sum=(0 0 0 0 0 0)
macro_f1_sum=(0 0 0 0 0 0)
micro_f1_sum=(0 0 0 0 0 0)
count=(0 0 0 0 0 0)
total_experiments=60  # 10 seeds × 2 settings × 3 contexts
current=0

get_index() {
  local setting=$1
  local context=$2

  if [[ "$setting" == "Setting2" && "$context" == "CONTEXT_all" ]]; then echo 0
  elif [[ "$setting" == "Setting2" && "$context" == "CONTEXT_sub" ]]; then echo 1
  elif [[ "$setting" == "Setting2" && "$context" == "CONTEXT_obj" ]]; then echo 2
  elif [[ "$setting" == "Setting3" && "$context" == "CONTEXT_all" ]]; then echo 3
  elif [[ "$setting" == "Setting3" && "$context" == "CONTEXT_sub" ]]; then echo 4
  elif [[ "$setting" == "Setting3" && "$context" == "CONTEXT_obj" ]]; then echo 5
  fi
}

update_metrics() {
  local idx=$1
  local accuracy=$2
  local macro_f1=$3
  local micro_f1=$4

  accuracy_sum[$idx]=$(echo "${accuracy_sum[$idx]} + $accuracy" | bc)
  macro_f1_sum[$idx]=$(echo "${macro_f1_sum[$idx]} + $macro_f1" | bc)
  micro_f1_sum[$idx]=$(echo "${micro_f1_sum[$idx]} + $micro_f1" | bc)
  count[$idx]=$(( ${count[$idx]} + 1 ))
}

print_averages() {
  local settings=("Setting2_CONTEXT_all" "Setting2_CONTEXT_sub" "Setting2_CONTEXT_obj" "Setting3_CONTEXT_all" "Setting3_CONTEXT_sub" "Setting3_CONTEXT_obj")
  
  echo -e "\nFinal Results:"
  for i in {0..5}; do
    if [[ ${count[$i]} -gt 0 ]]; then
      avg_accuracy=$(echo "${accuracy_sum[$i]} / ${count[$i]}" | bc -l)
      avg_macro_f1=$(echo "${macro_f1_sum[$i]} / ${count[$i]}" | bc -l)
      avg_micro_f1=$(echo "${micro_f1_sum[$i]} / ${count[$i]}" | bc -l)

      echo "---------------------------------------------"
      echo "Results for ${settings[$i]}"
      echo "---------------------------------------------"
      echo "Average Accuracy: $avg_accuracy"
      echo "Average Macro F1: $avg_macro_f1"
      echo "Average Micro F1: $avg_micro_f1"
    fi
  done
}

echo "Starting Setting2 experiments..."

for seed in {101..110}; do
  for context_group in CONTEXT_all CONTEXT_sub CONTEXT_obj; do
    ((current++))
    echo -n "Progress: $current/$total_experiments - Running Setting2 (seed=$seed, context=$context_group)... "
    
    output=$(python main.py --model_name LR --datadir ${DATADIR} --save_anno setting2_${context_group}_${seed} --dataname setting2-${seed} --context_column_group ${context_group} --random_seed ${seed} --penalty l1 --regularization 10 --max_iter 1000 --solver liblinear --class_num 3 --metrics accuracy,macro_f1,micro_f1)
    
    accuracy=$(echo "$output" | grep 'train results --' | grep -oE 'accuracy:[0-9.]+')
    macro_f1=$(echo "$output" | grep 'train results --' | grep -oE 'macro_f1:[0-9.]+')
    micro_f1=$(echo "$output" | grep 'train results --' | grep -oE 'micro_f1:[0-9.]+')
    accuracy=${accuracy#accuracy:}
    macro_f1=${macro_f1#macro_f1:}
    micro_f1=${micro_f1#micro_f1:}
    accuracy=${accuracy:-0}
    macro_f1=${macro_f1:-0}
    micro_f1=${micro_f1:-0}
    
    echo "done"
    
    idx=$(get_index "Setting2" "$context_group")
    update_metrics "$idx" "$accuracy" "$macro_f1" "$micro_f1"
  done
done

echo -e "\nStarting Setting3 experiments..."

for seed in {101..110}; do
  for context_group in CONTEXT_all CONTEXT_sub CONTEXT_obj; do
    ((current++))
    echo -n "Progress: $current/$total_experiments - Running Setting3 (seed=$seed, context=$context_group)... "
    
    output=$(python main.py --model_name LR --datadir ${DATADIR} --save_anno setting3_${context_group}_${seed} --dataname setting3-${seed} --context_column_group ${context_group} --random_seed ${seed} --penalty l1 --regularization 10 --max_iter 1000 --solver liblinear --class_num 3 --metrics accuracy,macro_f1,micro_f1)

    accuracy=$(echo "$output" | grep 'train results --' | grep -oE 'accuracy:[0-9.]+')
    macro_f1=$(echo "$output" | grep 'train results --' | grep -oE 'macro_f1:[0-9.]+')
    micro_f1=$(echo "$output" | grep 'train results --' | grep -oE 'micro_f1:[0-9.]+')

    accuracy=${accuracy#accuracy:}
    macro_f1=${macro_f1#macro_f1:}
    micro_f1=${micro_f1#micro_f1:}
    accuracy=${accuracy:-0}
    macro_f1=${macro_f1:-0}
    micro_f1=${micro_f1:-0}

    echo "done"
    
    idx=$(get_index "Setting3" "$context_group")
    update_metrics "$idx" "$accuracy" "$macro_f1" "$micro_f1"
  done
done

print_averages
