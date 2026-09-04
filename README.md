# 投篮生物力学分析系统（ShotOptix）

单文件 Streamlit 应用：篮球三分命中率多模态分析（弹道物理 + 出手反演 + 髋肩相位 + 高斯过程帕累托最优 + 半场热力图）。

## 本地运行

```bash
pip install -r requirements.txt
python shot_analyzer.py          # 或: streamlit run shot_analyzer.py
```

终端打印 `Local URL: http://localhost:8501` 即浏览器打开的网站地址。
无摄像头/视频时，打开页面“模拟数据演示”开关即可查看全部 9 个子图效果。

## 部署到 Streamlit Cloud（公网）

1. 把本仓库推送到 GitHub 公开仓库；
2. 打开 https://share.streamlit.io → 用 GitHub 登录 → New app；
3. 选仓库 / 分支 / 主文件 `shot_analyzer.py` → Deploy；
4. 得到公网链接（如 `xxx.streamlit.app`）。

> 云端可不装 `mediapipe`（体积大），脚本会自动降级为模拟姿态分析。
