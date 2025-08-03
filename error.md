# cached_download関数がhuggingface_hubからimportできない
原因：diffusers=0.20.2が古く、最新のhuggingface_hubと互換性がない

解決：huggingface_hubのバージョンを0.16.4に落として解決
```bash
pip install huggingface_hub==0.16.4
```

# CUDA kernel エラー (H200 NVL 未対応)
原因：PyTorch 2.0.1+cu117はCUDA capability sm_90 (H200)をサポートしていない

## 互換性チェック結果
✅ コード互換性: 問題なし (標準PyTorch APIのみ使用)
⚠️ 要確認: xformers==0.0.21, diffusers==0.20.2

## 推奨解決方法
```bash
# H200対応PyTorchに更新
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install xformers --index-url https://download.pytorch.org/whl/cu121
pip install "diffusers>=0.24.0"
```

# CustomDtype import エラー
原因：accelerate==0.16.0が古く、diffusers==0.34.0と互換性がない

解決：accelerateをアップデート
```bash
pip install "accelerate>=0.20.0"
```

# SiglipImageProcessor import エラー
原因：transformers==4.27.4が古く、diffusersの新機能と互換性がない

解決：transformersをアップデート
```bash
pip install "transformers>=4.35.0"
```

## Notebook Fixes

### XL-sliders-inference.ipynb
- importエラー解消: `randn_tensor` を削除し、`rescale_noise_cfg` を利用するよう修正
- `.safetensors` 対応: LoRA重みの読み込みに `safetensors.torch.load_file` ブランチを追加
- NoneTypeエラー解消: `_get_add_time_ids` に `text_encoder_2.config.projection_dim` を渡すよう修正
- デバッグ用アサーション追加: `original_size` と `target_size` の値を確認するための `assert` を挿入

### train-flux-concept-sliders.ipynb
- LoRA読み込み箇所を有効化: コメントアウトされた `load_state_dict` 部分をアンコメントし、`.pt` と `.safetensors` 両対応を追加予定

