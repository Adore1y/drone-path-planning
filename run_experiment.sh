#!/bin/bash

# 无人机路径规划实验启动脚本

# 显示彩色输出
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${BLUE}==========================================${NC}"
echo -e "${BLUE}   无人机路径规划与比较实验启动脚本   ${NC}"
echo -e "${BLUE}==========================================${NC}"

# 确保已安装所需依赖
check_dependencies() {
  echo -e "\n${YELLOW}正在检查依赖项...${NC}"
  if ! command -v python3 &> /dev/null; then
    echo -e "${RED}错误: 未找到 Python3。请先安装 Python3。${NC}"
    exit 1
  fi
  
  # 检查 Webots
  if ! [ -d "/Applications/Webots.app" ]; then
    echo -e "${RED}警告: 未在默认位置找到 Webots。如果已安装到其他位置，请忽略此警告。${NC}"
  else
    echo -e "${GREEN}Webots 已安装。${NC}"
  fi
  
  # 安装 Python 依赖
  echo -e "${YELLOW}正在安装 Python 依赖项...${NC}"
  pip install -r webots/controllers/mavic_python/requirements.txt
  
  echo -e "${GREEN}依赖项检查完成。${NC}"
}

# 生成新的障碍物和路径点
generate_obstacles() {
  echo -e "\n${YELLOW}是否要生成新的障碍物和路径点? [y/n]${NC}"
  read -r generate_new
  
  if [[ $generate_new == "y" || $generate_new == "Y" ]]; then
    echo -e "${YELLOW}生成新的障碍物和路径点...${NC}"
    cd scripts && python generate_obstacles.py && cd ..
    
    # 更新世界文件
    echo -e "${YELLOW}更新世界文件...${NC}"
    # 找到世界文件的最后一行（应该是一个单独的右花括号）
    last_line=$(grep -n '^}$' webots/worlds/mixed_scenario.wbt | tail -1 | cut -d':' -f1)
    
    if [ -n "$last_line" ]; then
      # 删除最后一行，追加新的障碍物和路径点，然后添加闭合的右花括号
      sed -i.bak "${last_line}d" webots/worlds/mixed_scenario.wbt
      cat scripts/obstacles_and_waypoints.wbt >> webots/worlds/mixed_scenario.wbt
      echo "}" >> webots/worlds/mixed_scenario.wbt
      echo -e "${GREEN}世界文件已更新。${NC}"
    else
      echo -e "${RED}错误: 无法更新世界文件。${NC}"
    fi
  else
    echo -e "${BLUE}使用现有的障碍物和路径点。${NC}"
  fi
}

# 运行 Webots 仿真
run_simulation() {
  echo -e "\n${YELLOW}是否要立即启动 Webots 仿真? [y/n]${NC}"
  read -r start_sim
  
  if [[ $start_sim == "y" || $start_sim == "Y" ]]; then
    echo -e "${YELLOW}启动 Webots 仿真...${NC}"
    open -a Webots.app "$(pwd)/webots/worlds/mixed_scenario.wbt"
    echo -e "${GREEN}Webots 已启动。${NC}"
    
    echo -e "\n${BLUE}提示: ${NC}"
    echo -e "1. 点击界面顶部的 ▶️ 按钮开始仿真"
    echo -e "2. 仿真完成后，运行分析脚本查看结果"
  else
    echo -e "${BLUE}稍后可以手动启动 Webots 仿真。${NC}"
  fi
}

# 生成深度学习模拟路径
generate_dl_path() {
  echo -e "\n${YELLOW}是否要生成深度学习模拟路径进行比较? [y/n]${NC}"
  read -r gen_dl
  
  if [[ $gen_dl == "y" || $gen_dl == "Y" ]]; then
    echo -e "${YELLOW}选择深度学习模型类型:${NC}"
    echo -e "1) DRL (深度强化学习)"
    echo -e "2) CNN (卷积神经网络)"
    echo -e "3) LSTM (长短期记忆网络)"
    echo -e "4) Hybrid (混合模型)"
    read -r model_choice
    
    case $model_choice in
      1) model_type="drl" ;;
      2) model_type="cnn" ;;
      3) model_type="lstm" ;;
      4) model_type="hybrid" ;;
      *) model_type="drl" ;;
    esac
    
    echo -e "${YELLOW}生成 ${model_type} 模型路径...${NC}"
    cd webots/controllers/mavic_python && python generate_dl_path.py --model_type ${model_type}
    cd ../../..
    
    echo -e "${GREEN}深度学习路径已生成。${NC}"
  else
    echo -e "${BLUE}跳过生成深度学习路径。${NC}"
  fi
}

# 分析结果
analyze_results() {
  echo -e "\n${YELLOW}是否要分析飞行数据? [y/n]${NC}"
  read -r analyze
  
  if [[ $analyze == "y" || $analyze == "Y" ]]; then
    echo -e "\n${YELLOW}请选择分析类型:${NC}"
    echo -e "1) 基本数据分析 (analyze_flight_data.py)"
    echo -e "2) 路径点导航分析和对比 (analyze_waypoints.py)"
    read -r analysis_type
    
    if [[ $analysis_type == "1" ]]; then
      echo -e "${YELLOW}执行基本数据分析...${NC}"
      cd webots/controllers/mavic_python && python analyze_flight_data.py
      cd ../../..
    else
      echo -e "${YELLOW}执行路径点导航分析和对比...${NC}"
      cd webots/controllers/mavic_python && python analyze_waypoints.py
      cd ../../..
    fi
    
    echo -e "${GREEN}分析完成。${NC}"
    echo -e "${BLUE}结果保存在 webots/controllers/mavic_python/flight_data/ 目录中。${NC}"
  else
    echo -e "${BLUE}跳过分析。${NC}"
  fi
}

# 主流程
main() {
  check_dependencies
  generate_obstacles
  run_simulation
  
  echo -e "\n${YELLOW}按任意键继续进行深度学习路径生成和分析 (或 Ctrl+C 退出)...${NC}"
  read -n 1 -s
  
  generate_dl_path
  analyze_results
  
  echo -e "\n${GREEN}实验流程完成!${NC}"
  echo -e "${BLUE}如需重新运行实验，请再次执行此脚本。${NC}"
}

# 运行主函数
main 