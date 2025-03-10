#!/bin/bash
# 运行GAT-TD3算法训练脚本

# 当前目录
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
cd "$SCRIPT_DIR/../.." || exit

# 默认参数
TIMESTEPS=100000
ENV_TYPE="urban"
CONFIG_FILE="algorithms/drl/configs/gat_td3_config.json"
ENV_CONFIG_FILE="algorithms/drl/configs/env_config.json"
OUTPUT_DIR="training_results"
HEADLESS=false
NO_NORMALIZE=false
NO_GRAPH=false
NO_ENERGY=false
LOAD_MODEL=""
EVAL_ONLY=false
EVAL_EPISODES=10
RENDER=false
SEED=""

# 打印帮助
print_help() {
    echo "GAT-TD3 无人机能量高效路径规划训练脚本"
    echo ""
    echo "使用方法:"
    echo "  ./run_gat_td3.sh [选项]"
    echo ""
    echo "选项:"
    echo "  -t, --timesteps N   训练总步数 (默认: 100000)"
    echo "  -e, --env TYPE      环境类型 (默认: urban)"
    echo "  -c, --config FILE   GAT-TD3配置文件路径"
    echo "  --env-config FILE   环境配置文件路径"
    echo "  -o, --output DIR    输出目录 (默认: training_results)"
    echo "  -s, --seed N        随机种子"
    echo "  -h, --headless      使用无头模式运行Webots"
    echo "  -n, --no-normalize  不对观测进行归一化"
    echo "  -g, --no-graph      不使用图表示"
    echo "  -x, --no-energy     不使用能量模型"
    echo "  -l, --load FILE     加载已训练的模型"
    echo "  -v, --eval-only     仅进行评估，不训练"
    echo "  -p, --episodes N    评估时的情节数 (默认: 10)"
    echo "  -r, --render        渲染评估过程"
    echo "  --help              显示此帮助信息"
    echo ""
    echo "示例:"
    echo "  ./run_gat_td3.sh -t 500000 -e urban -h     # 在无头模式下训练50万步"
    echo "  ./run_gat_td3.sh -v -l models/gattd3       # 评估已训练的模型"
}

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case $1 in
        -t|--timesteps)
            TIMESTEPS="$2"
            shift 2
            ;;
        -e|--env)
            ENV_TYPE="$2"
            shift 2
            ;;
        -c|--config)
            CONFIG_FILE="$2"
            shift 2
            ;;
        --env-config)
            ENV_CONFIG_FILE="$2"
            shift 2
            ;;
        -o|--output)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        -s|--seed)
            SEED="$2"
            shift 2
            ;;
        -h|--headless)
            HEADLESS=true
            shift
            ;;
        -n|--no-normalize)
            NO_NORMALIZE=true
            shift
            ;;
        -g|--no-graph)
            NO_GRAPH=true
            shift
            ;;
        -x|--no-energy)
            NO_ENERGY=true
            shift
            ;;
        -l|--load)
            LOAD_MODEL="$2"
            shift 2
            ;;
        -v|--eval-only)
            EVAL_ONLY=true
            shift
            ;;
        -p|--episodes)
            EVAL_EPISODES="$2"
            shift 2
            ;;
        -r|--render)
            RENDER=true
            shift
            ;;
        --help)
            print_help
            exit 0
            ;;
        *)
            echo "未知选项: $1"
            print_help
            exit 1
            ;;
    esac
done

# 构建命令
CMD="python algorithms/drl/run_gat_td3.py -t $TIMESTEPS -e $ENV_TYPE -c $CONFIG_FILE --env-config $ENV_CONFIG_FILE -o $OUTPUT_DIR"

# 添加可选参数
if [ "$HEADLESS" = true ]; then
    CMD="$CMD --headless"
fi

if [ "$NO_NORMALIZE" = true ]; then
    CMD="$CMD --no-normalize"
fi

if [ "$NO_GRAPH" = true ]; then
    CMD="$CMD --no-graph"
fi

if [ "$NO_ENERGY" = true ]; then
    CMD="$CMD --no-energy"
fi

if [ -n "$LOAD_MODEL" ]; then
    CMD="$CMD --load $LOAD_MODEL"
fi

if [ "$EVAL_ONLY" = true ]; then
    CMD="$CMD --eval-only"
fi

if [ "$RENDER" = true ]; then
    CMD="$CMD --render"
fi

if [ -n "$SEED" ]; then
    CMD="$CMD -s $SEED"
fi

CMD="$CMD --eval-episodes $EVAL_EPISODES"

# 打印命令
echo "执行命令: $CMD"
echo "========================================="

# 执行命令
eval "$CMD"

# 检查执行结果
if [ $? -eq 0 ]; then
    echo "========================================="
    echo "训练完成！"
    echo "结果保存到: $OUTPUT_DIR"
    echo "========================================="
else
    echo "========================================="
    echo "训练失败，错误代码: $?"
    echo "========================================="
fi 