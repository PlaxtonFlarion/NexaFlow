#!/bin/bash

# === 定位当前脚本所在目录 ===
ROOT="$(cd "$(dirname "$0")" && pwd)"

# === 配置区域 ===
APP="Framix"
SRC="$ROOT/applications"
DST="$ROOT/dist"
DMG="$APP-macos-v1.0.0.dmg"
VOL="$APP Installer"

# === 自动匹配背景图 ===
BGP=$(find "$ROOT/schematic/resources/images/macos" -type f -name '*macos_bg.png' | head -n 1)

if [ -z "$BGP" ]; then
  echo "❌ 未找到匹配的 macos_bg.png 背景图"
  exit 1
fi

# === 检查依赖 ===
if ! command -v brew &>/dev/null; then
  echo "❌ 未检测到 Homebrew，请先安装："
  echo '/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"'
  exit 1
fi

if ! command -v create-dmg &>/dev/null; then
  echo "❌ 未检测到 create-dmg，你可以通过以下命令安装："
  echo "brew install create-dmg"
  exit 1
fi

# === 校验路径 ===
if [ ! -d "$SRC" ]; then
  echo "❌ 应用目录不存在：$SRC"
  exit 1
fi

if [ ! -f "$BGP" ]; then
  echo "❌ 背景图不存在：$BGP"
  exit 1
fi

# === 删除 applications 目录下所有 .dist / .build 结尾的目录 ===
echo "🧹 清理 $SRC 下的 .dist / .build 目录..."
find "$SRC" -type d \( -name "*.dist" -o -name "*.build" \) -exec rm -rf {} +

# === 删除旧的 dist 目录（包括所有文件）===
rm -rf "$ROOT/dist"

# === 重新创建空的 dist 目录 ===
mkdir -p "$ROOT/dist"

# === 执行 create-dmg（v1.2.1语法） ===
create-dmg \
  --volname "$VOL" \
  --window-size 500 350 \
  --icon-size 100 \
  --background "$BGP" \
  --icon "$APP.app" 125 175 \
  --app-drop-link 375 175 \
  --format UDZO \
  --no-internet-enable \
  "$DST/$DMG" \
  "$SRC"

# === 成功提示 ===
echo "✅ DMG 已生成：$DST/$DMG"

# === 生成 SHA256 校验文件 ===
echo "🔐 正在生成 SHA256 校验..."
FILENAME=$(basename "$DMG")
shasum -a 256 "$DST/$DMG" | sed "s|$DST/||" > "$DST/$FILENAME.sha256"
echo "✅ SHA256 已写入：$DST/$FILENAME.sha256"

# === 自动打开输出目录 ===
#open "$DST"
