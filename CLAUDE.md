# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 專案概觀

SkillOpt 的精簡重現版:一種「文字空間優化器」,**不微調權重**,而是為一個凍結的
LLM agent 訓練出一份可重複使用的自然語言技能(`best_skill.md`)。整套流程仿照神經
網路訓練——epoch、mini-batch rollout,加上一道**驗證閘門**:只有候選技能在保留的
val 集上指標提升時才接受該次編輯。詳見 `README.md`。

## 常用指令

```bash
# 安裝(務必用 uv,專案規範禁止 pip)
uv venv && uv pip install -e ".[dev]"
cp .env.example .env                 # 本機服務通常 OPENAI_API_KEY 任意值即可

# 下載 HotpotQA 切分 -> data/hotpotqa/{train,val,test}/items.json
uv run skillopt-download --out data/hotpotqa --n-train 64 --n-val 64 --n-test 200

# 訓練(產物在 outputs/<run_name>/);CLI 旗標會覆寫 YAML 欄位
uv run skillopt-train --config configs/hotpotqa/default.yaml \
    --split-dir data/hotpotqa --out-root outputs
# 常用覆寫:--target-model --base-url --num-epochs --batch-size --val-size
#          --workers --metric {em,f1} --no-test

# 測試(全部以 fake LLM 離線執行,完全不需網路/不需服務)
uv run pytest
uv run pytest tests/test_trainer_functional.py::test_fit_accepts_improving_edit  # 單一測試
```

## 執行前提

訓練/下載需要一個 **OpenAI 相容 chat endpoint**(預設指向本機 vLLM,
`configs/hotpotqa/default.yaml` 的 `base_url=http://localhost:8080/v1`)。關鍵點:
`model.target_model` 與 `model.optimizer_model` 必須等於服務 `/v1/models` 回報的
**served-model-name**(用 `curl http://localhost:8080/v1/models` 確認),否則呼叫會
失敗。**測試不需要服務**——它們注入 fake LLM。

## 架構:資料流

一次訓練步驟(`Trainer.fit`,`skillopt/trainer.py`)串起所有模組:

1. `agent.run_batch` 用**目前最佳技能**在一個 train mini-batch 上平行 rollout
   (`ThreadPoolExecutor`,因 API 呼叫是 IO-bound),產出 `Trajectory` 清單。
2. `evaluator.evaluate` 就地把每條軌跡標記 `correct`(HotpotQA 風格的正規化 EM)。
3. `optimizer.propose_skill` 把成功+失敗軌跡當證據,請 optimizer LLM 提出**一次有界
   編輯**,回傳候選技能字串。
4. **驗證閘門**:候選技能在 val 集上重跑;唯有 `cand_metric > best + min_improvement`
   才接受,成為新的 `best_skill`。

關鍵不變量與設計:
- **凍結 agent**:`agent.py` 的模型從不被訓練;唯一可變輸入是注入 system prompt 的
  技能文字(`BASE_INSTRUCTIONS` + `# Skill\n{skill}`)。部署只需把 `best_skill.md`
  接到任一 agent 的 system prompt 前。
- **optimizer 穩定機制**(`optimizer.py`):同時呈現對/錯軌跡(足夠證據)、要求小編輯
  (有界更新)、`OptimizerState.rejected_edits` 記住最近 3 個被拒版本以避免重複提案、
  每步僅一次需過閘門的編輯(慢更新)。**接受編輯後拒絕記憶會清空**。
- **提早停止**:連續 `patience` 次拒絕即停。
- **prompt 字串刻意保留英文**(`SEED_SKILL`、`_OPTIMIZER_SYSTEM`、`BASE_INSTRUCTIONS`),
  對模型推理較穩定——即使本專案規範要求中文回覆,這些字串不要翻成中文。
- **reasoning 模型**:`enable_thinking=false` 經 vLLM `chat_template_kwargs` 關閉思考以
  大幅加速;agent 只讀最終 `content`,不讀思考內容。

## 設定機制

`Config`(`skillopt/config.py`)從 YAML 載入並含 `ModelConfig` / `TrainConfig` 兩個巢狀
區段。`apply_overrides` 接受**扁平**的 CLI 覆寫字典,依欄位名自動導向正確的巢狀區段
(未知 key 會 raise)。`scripts/train.py` 即用此把 argparse 旗標映射進設定。

## 產物

`outputs/<run_name>/`:`best_skill.md`(唯一可部署產物)、`history.json`(每步指標與
接受/拒絕)、`skills/skill_vXXXX_{seed,accept,reject}.md`、`steps/step_XXXX.json`、
`test_result.json`(保留 test 集的最終 EM/F1)。
