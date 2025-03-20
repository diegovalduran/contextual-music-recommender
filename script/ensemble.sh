#!/bin/bash

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "${SCRIPT_DIR}/../src"
DATADIR="../datasets/Psychological_Datasets/"

# Initialize arrays and counters
declare -a accuracy_sum=(0 0 0 0 0 0)
declare -a macro_f1_sum=(0 0 0 0 0 0)
declare -a micro_f1_sum=(0 0 0 0 0 0)
declare -a count=(0 0 0 0 0 0)
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

is_numeric() {
  [[ $1 =~ ^[0-9]+(\.[0-9]+)?$ ]]
}

validate_metric() {
  local metric=$1
  if is_numeric "$metric" && (( $(echo "$metric >= 0" | bc -l) )) && (( $(echo "$metric <= 1" | bc -l) )); then
    echo "$metric"
  else
    echo "0" 
  fi
}

update_metrics() {
  local idx=$1
  local accuracy=$2
  local macro_f1=$3
  local micro_f1=$4

  accuracy_sum[$idx]=$(echo "${accuracy_sum[$idx]} + $accuracy" | bc -l)
  macro_f1_sum[$idx]=$(echo "${macro_f1_sum[$idx]} + $macro_f1" | bc -l)
  micro_f1_sum[$idx]=$(echo "${micro_f1_sum[$idx]} + $micro_f1" | bc -l)
  count[$idx]=$(( ${count[$idx]} + 1 ))
}

print_averages() {
  local settings=("Setting2_CONTEXT_all" "Setting2_CONTEXT_sub" "Setting2_CONTEXT_obj" "Setting3_CONTEXT_all" "Setting3_CONTEXT_sub" "Setting3_CONTEXT_obj")
  
  echo -e "\nFinal Results:"
  for i in {0..5}; do
    if [[ ${count[$i]} -gt 0 ]]; then
      avg_accuracy=$(echo "scale=4; ${accuracy_sum[$i]} / ${count[$i]}" | bc -l)
      avg_macro_f1=$(echo "scale=4; ${macro_f1_sum[$i]} / ${count[$i]}" | bc -l)
      avg_micro_f1=$(echo "scale=4; ${micro_f1_sum[$i]} / ${count[$i]}" | bc -l)
      echo "---------------------------------------------"
      echo "Average Results for ${settings[$i]}"
      echo "---------------------------------------------"
      echo "Average Accuracy: $avg_accuracy"
      echo "Average Macro F1: $avg_macro_f1"
      echo "Average Micro F1: $avg_micro_f1"
      echo ""
    fi
  done
}

echo "Starting ensemble experiments with stacking classifier..."

for seed in {101..110}; do
  for setting in "Setting2" "Setting3"; do
    for context_group in CONTEXT_all CONTEXT_sub CONTEXT_obj; do
      ((current++))
      echo -n "Progress: $current/$total_experiments - Running $setting (seed=$seed, context=$context_group)... "
      
      output=$(python ensemble.py \
        --datadir ${DATADIR} \
        --model_type stacking \
        --dataname ${setting}-${seed} \
        --context_column_group ${context_group})
      
      # Validate and extract metrics
      accuracy=$(validate_metric $(echo "$output" | grep -m1 "Accuracy   :" | sed -E 's/[^0-9.]*//g' || echo "0"))
      macro_f1=$(validate_metric $(echo "$output" | grep -m1 "Macro F1   :" | sed -E 's/[^0-9.]*//g' || echo "0"))
      micro_f1=$(validate_metric $(echo "$output" | grep -m1 "Micro F1   :" | sed -E 's/[^0-9.]*//g' || echo "0"))
      
      echo "done"
      
      # Update metrics if valid
      if (( $(echo "$accuracy > 0" | bc -l) )); then
        idx=$(get_index "$setting" "$context_group")
        update_metrics "$idx" "$accuracy" "$macro_f1" "$micro_f1"
      else
        echo "Warning: Invalid metrics detected for $setting, seed=$seed, context=$context_group"
      fi
    done
  done
done

print_averages
