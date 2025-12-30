import streamlit as st
import pandas as pd
import numpy as np
import torch
from torch import nn, optim
import io
import csv
from pulp import *

# === スタイル調整 ===
st.markdown("""
<style>
.main { padding: 1rem 2rem !important; margin: 0rem !important; }
.block-container { padding: 1.5rem 2rem !important; max-width: 100vw !important; }
header, footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# === 定数定義 ===
category_options = ["日", "E3", "長", "N4", "夜", "休"]
nurses = list("ABCDEFGHIJKLMNOPQRST")
dates1 = []
for date in range(21,32):
    dates1.append(date)
dates2 = []
for date in range(1,21):
    dates2.append(date)
dates = dates1 + dates2 #J
shifts = [0,1,2,3,4,5]
input_dim = output_dim = len(nurses) * len(dates) * len(shifts)
hidden_dim = hidden2_dim = hidden3_dim = 30
eta = 0.0005

# === NN定義 ===
class MyModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, hidden2_dim, hidden3_dim):
        super(MyModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden2_dim)
        self.fc3 = nn.Linear(hidden2_dim, hidden3_dim)
        self.fc4 = nn.Linear(hidden3_dim, output_dim)
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        return torch.sigmoid(self.fc4(x))

# === 関数定義 ===
def preprocess(sample_file):
    return pd.read_csv(sample_file)

def convert_to_csv(df):
    return df.to_csv().encode('shift_jis')

def df_to_schedule(df):
    matrix = df.values.tolist()
    for i in range(len(matrix)):
        del matrix[i][0]
    schedules = []
    for i in range(len(matrix)):
        schedule = []
        for j in range(len(dates)):
            val = matrix[i][j]
            sche = [1,0,0,0,0,0] if val=="日" else [0,1,0,0,0,0] if val=="E3" else [0,0,1,0,0,0] if val=="長" else [0,0,0,1,0,0] if val=="N4" else [0,0,0,0,1,0] if val=="夜" else [0,0,0,0,0,1]
            schedule.append(sche)
        schedules.append(schedule)
    return schedules

def schedule_to_df(schedule):
    df_matrix = []
    for i in range(len(schedule)):
        works = []
        for j in range(len(schedule[0])):
            shift = schedule[i][j]
            idx = np.argmax(shift)
            works.append(["日", "E3", "長", "N4", "夜", "休"][idx])
        df_matrix.append(works)
    for i in range(len(df_matrix)):
        df_matrix[i].insert(0, nurses[i])
    columns = ["名前"] + [f"{d}日" for d in dates]
    return pd.DataFrame(df_matrix, columns=columns)

def train_step(x_tensor, y_tensor):
    model = st.session_state.model
    optimizer = st.session_state.optimizer
    criterion = nn.MSELoss()
    model.train()
    x_flat = x_tensor.view(x_tensor.size(0), -1)
    y_flat = y_tensor.view(y_tensor.size(0), -1)
    y_pred = model(x_flat)
    loss = criterion(y_pred, y_flat)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    st.session_state.losses.append(loss.item())
    return y_pred.view(x_tensor.shape)

def calc_mse(schedule1, schedule2):
    return np.mean(np.square(np.array(schedule1).flatten() - np.array(schedule2).flatten()))

def session_init():
    st.session_state.model = MyModel(input_dim, hidden_dim, output_dim, hidden2_dim, hidden3_dim)
    st.session_state.optimizer = optim.Adam(st.session_state.model.parameters(), lr=eta)
    st.session_state.losses = []
    st.session_state.current_schedule = None
    st.session_state.edited_schedule = None
    st.session_state.learning_finished = False
    st.session_state.re_learning_finished = False
    st.session_state.testing_started = False
    st.session_state.last_change = None
    st.session_state.last_mse = None


# === セッション初期化 ===
if "model" not in st.session_state:
    session_init()

# === UI ===
st.title("修正シフト逐次学習システムー続き学習対応版ー")

# === モデルテスト ===
st.markdown("---")
st.header("確定済みモデルのテスト1")

model_file = st.file_uploader("モデル (.pt) をアップロード", type="pt")
test_file = st.file_uploader("シフトCSVをアップロード", type="csv")

# ✅ モデルとデータがアップロードされたらテスト開始
if model_file and test_file:
    st.session_state.testing_started = True
    st.session_state.testing_completed = True  

    # ✅ アップロードした学習済みモデルを session_state.model にロード
    buffer = io.BytesIO(model_file.read())
    st.session_state.model.load_state_dict(torch.load(buffer, map_location="cpu"))
    st.session_state.model.eval()

    # === 入力CSVを読み込み ===
    test_df = preprocess(test_file)
    test_schedule = df_to_schedule(test_df)
    x_tensor = torch.tensor(test_schedule, dtype=torch.float32).unsqueeze(0).view(1, -1)

    # === モデル推論 ===
    with torch.no_grad():
        output_tensor = st.session_state.model(x_tensor)

    predicted_schedule = output_tensor.view(20, 31, 6).tolist()
    st.session_state.current_schedule = predicted_schedule

    mse = calc_mse(test_schedule, predicted_schedule)
    st.write(f"モデル出力とテストシフトのMSE: {mse:.4f}")

    result_df = schedule_to_df(predicted_schedule)
    st.subheader("モデルによって生成されたシフト")
    st.dataframe(result_df)

    result_csv = convert_to_csv(result_df)
    st.download_button(
        "生成されたシフトをダウンロードする",
        result_csv,
        "predicted_shift_1.csv",
        "text/csv",
        key="download-pred"
    )

# === テスト後の再修正・再学習 ===
if st.session_state.testing_started:
    st.markdown("---")
    st.header("モデル出力の再修正と繰り返し学習1")

    # 🔽 毎回最新 current_schedule から表示用 DataFrame を再生成
    re_df_display = schedule_to_df(st.session_state.current_schedule)

    # 🔽 各列の編集選択肢
    re_column_config = {
        col: st.column_config.SelectboxColumn(label=col, options=category_options, required=True)
        for col in re_df_display.columns if col != "名前"
    }

    # 🔽 editorの key を動的に（学習回数によって） 
    re_edited_df = st.data_editor(
        re_df_display,
        column_config=re_column_config,
        use_container_width=True,
        num_rows="dynamic",
        key=f"re_editor_{len(st.session_state.losses)}"
    )

    # 🔽 学習ボタン
    if not st.session_state.re_learning_finished:
        if st.button("この再修正を学習する"):
            # データ変換と学習
            re_edited_schedule = df_to_schedule(re_edited_df)
            st.session_state.edited_schedule = re_edited_schedule

            x_tensor = torch.tensor(st.session_state.current_schedule, dtype=torch.float32).unsqueeze(0)
            y_tensor = torch.tensor(re_edited_schedule, dtype=torch.float32).unsqueeze(0)
            before = st.session_state.model.fc4.weight.clone().detach()
            new_schedule_tensor = train_step(x_tensor, y_tensor)

            # 🔽 モデル出力をセッションに反映
            st.session_state.current_schedule = new_schedule_tensor.squeeze(0).tolist()

            after = st.session_state.model.fc4.weight.clone().detach()

            st.session_state.last_change = torch.norm(after - before).item()

            st.write("変化量:", torch.norm(after - before).item())

            # MSE表示
            mse = calc_mse(st.session_state.current_schedule, re_edited_schedule)
            st.session_state.last_mse = mse
            st.success(f"MSE: {mse:.4f}")

            st.rerun()

    if st.session_state.last_change is not None:
        st.info(f"前回学習時の変化量: {st.session_state.last_change:.6f}")

    if st.session_state.last_mse is not None:
        st.info(f"前回学習時の MSE: {st.session_state.last_mse:.4f}")


    # 🔽 モデル保存
    if st.button("再学習を終了してモデルを確定する"):
        st.session_state.re_learning_finished = True
        torch.save(st.session_state.model.state_dict(), "retrained_model_1.pt")
        st.success("再学習済みモデルを保存しました。これ以降は学習されません。")

    # 🔽 学習済みの案内表示
    if st.session_state.re_learning_finished:
        st.info("✅ このモデルは確定されています。再修正・再学習は無効です。")

    # 🔽 損失関数の推移表示
    if st.button("損失関数の推移を表示（Show Loss Curve）"):
        st.line_chart(st.session_state.losses)

        # === 損失関数CSVの書き出し ===
        loss_df = pd.DataFrame({
            "Epoch": range(1, len(st.session_state.losses) + 1),
            "Loss": st.session_state.losses
        })
        loss_csv = loss_df.to_csv(index=False).encode("utf-8-sig")

        st.download_button(
            label="損失関数の値をCSVとして保存（Download Loss CSV）",
            data=loss_csv,
            file_name="loss_history.csv",
            mime="text/csv"
        )

    # 🔽 現在のスケジュール出力を再生成＆ダウンロード
    re_csv = convert_to_csv(schedule_to_df(st.session_state.current_schedule))
    st.download_button(
        "現在の修正済みシフトをダウンロードする",
        re_csv,
        "retrained_schedule_1.csv",
        "text/csv",
        key="download-retrained"
    )