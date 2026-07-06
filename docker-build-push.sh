#!/bin/bash
# PocketGraphRAG 多架构 Docker 镜像构建脚本
# 对标 LightRAG docker-build-push.sh，支持 amd64 + arm64 双架构
#
# 使用方式：
#   ./docker-build-push.sh                    # 本地构建
#   ./docker-build-push.sh --push             # 构建并推送到 registry
#   REGISTRY=ghcr.io/jyzisliubi ./docker-build-push.sh --push
#
# 需要：
#   docker buildx（Docker 19.03+ 自带）
#   docker login（推送时需要）

set -e

# 配置
REGISTRY="${REGISTRY:-pocketgraphrag}"
IMAGE_NAME="${IMAGE_NAME:-pocketgraphrag}"
VERSION="${VERSION:-0.3.7}"
PLATFORMS="${PLATFORMS:-linux/amd64,linux/arm64}"

# 颜色输出
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}PocketGraphRAG Docker Build${NC}"
echo -e "${BLUE}Image: ${REGISTRY}/${IMAGE_NAME}:${VERSION}${NC}"
echo -e "${BLUE}Platforms: ${PLATFORMS}${NC}"
echo -e "${BLUE}========================================${NC}"

# 检查 buildx
if ! docker buildx version > /dev/null 2>&1; then
    echo -e "${YELLOW}Warning: docker buildx not available, falling back to single-arch build${NC}"
    docker build -t "${REGISTRY}/${IMAGE_NAME}:${VERSION}" .
    docker tag "${REGISTRY}/${IMAGE_NAME}:${VERSION}" "${REGISTRY}/${IMAGE_NAME}:latest"
    echo -e "${GREEN}Build complete: ${REGISTRY}/${IMAGE_NAME}:${VERSION}${NC}"
    exit 0
fi

# 创建 builder（如果不存在）
BUILDER_NAME="pocketgraphrag-builder"
if ! docker buildx inspect "$BUILDER_NAME" > /dev/null 2>&1; then
    echo -e "${BLUE}Creating buildx builder: ${BUILDER_NAME}${NC}"
    docker buildx create --name "$BUILDER_NAME" --use
fi
docker buildx use "$BUILDER_NAME"

# 构建参数
TAGS="--tag ${REGISTRY}/${IMAGE_NAME}:${VERSION} --tag ${REGISTRY}/${IMAGE_NAME}:latest"

# 推送
if [ "$1" = "--push" ]; then
    echo -e "${BLUE}Building and pushing multi-arch images...${NC}"
    docker buildx build \
        --platform "${PLATFORMS}" \
        ${TAGS} \
        --push \
        .
    echo -e "${GREEN}Pushed: ${REGISTRY}/${IMAGE_NAME}:${VERSION} (${PLATFORMS})${NC}"
else
    echo -e "${BLUE}Building multi-arch images (local only)...${NC}"
    # 本地构建仅支持单架构（buildx 限制）
    docker buildx build \
        --platform "linux/amd64" \
        ${TAGS} \
        --load \
        .
    echo -e "${GREEN}Built: ${REGISTRY}/${IMAGE_NAME}:${VERSION} (linux/amd64)${NC}"
fi

# 验证
echo -e "${BLUE}Verifying image...${NC}"
docker run --rm "${REGISTRY}/${IMAGE_NAME}:${VERSION}" python -c "
from PocketGraphRAG.config import WORKSPACE
print(f'PocketGraphRAG v0.3.7, WORKSPACE={WORKSPACE}')
print('Image verification passed!')
"

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}Build complete!${NC}"
echo -e "${GREEN}  Image: ${REGISTRY}/${IMAGE_NAME}:${VERSION}${NC}"
echo -e "${GREEN}  Latest: ${REGISTRY}/${IMAGE_NAME}:latest${NC}"
echo -e "${GREEN}========================================${NC}"
