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
eta = 0.01

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

def fix_shift(schedule, nurses, dates, shifts):
    times = [7.5,7.5,11.0,7.5,11.25,0]
    w_iota = []
    w_kappa = []   #重みを大きく
    w_alpha = []
    for j in range(len(dates)):
        w_iota.append(1)
        w_kappa.append(1000)
        w_alpha.append(1000)
        
    w_lambda = []
    for i in range(len(nurses)):
        w_lambda.append(1)

    w_u = []   #夜勤の回数に対する重み
    for i in range(len(nurses)):
        w = []
        for i_dash in range(len(nurses)):
            w.append(10)
        w_u.append(w)

    L = [5,5,4,4] #各クールの長さ

    #モデルの作成
    model = LpProblem("Nurse_Scheduling_Problem_for_Bantane_Hospital", LpMinimize)
    
    #変数
    x = {} #決定変数
    h = {}
    for i in range(len(nurses)):
        for j in range(len(dates)):
            for k in range(len(shifts)):
                x[i,j,k] = LpVariable("x{}-{}-{}".format(i,j,k), cat=LpBinary)
                h[i,j,k] = LpVariable("h{}-{}-{}".format(i,j,k), lowBound=0)

    y = {}
    z = {}
    v = {}
    for i in range(len(nurses)):
        for j in range(len(dates)):
            y[i,j] = LpVariable("y{}-{}".format(i,j), cat=LpBinary)
            z[i,j] = LpVariable("z{}-{}".format(i,j), cat=LpBinary)
            v[i,j] = LpVariable("v{}-{}".format(i,j), cat=LpBinary)

    iota = {}
    e_3 = {}
    #kappa = {}
    #alpha = {}
    for j in range(len(dates)):
        iota[j] = LpVariable("iota{}".format(j), lowBound=0, upBound=3)
        e_3[j] = LpVariable("e_3{}".format(j), lowBound=0, upBound=3)
        #kappa[j] = LpVariable("kappa{}".format(j), lowBound=0, upBound=1)
        #alpha[j] = LpVariable("alpha{}".format(j), lowBound=0, upBound=1)

    lamb = {} #lambda
    add = {}
    for i in range(len(nurses)):
        lamb[i] = LpVariable("lamb{}".format(i), lowBound=0, upBound=1)
        add[i] = LpVariable("add{}".format(i), lowBound=0)
    
    u = {}
    for i in range(len(nurses)):
        for i_dash in range(len(nurses)):
            u[i,i_dash] = LpVariable("u{}-{}".format(i,i_dash), lowBound=0, upBound=3)

    a = {}  #クールを実現させる変数
    for i in range(len(nurses)):
        for j in range(len(dates)):
            for p in range(0,4):
                a[i,j,p] = LpVariable("a{}-{}-{}".format(i,j,p), cat=LpBinary)

    #目的関数
    model += 1000*lpSum([lpSum([lpSum([h[i,j,k] for k in range(len(shifts))]) for j in range(len(dates))]) for i in range(len(nurses))]) - lpSum([lpSum([y[i,j]+z[i,j]-v[i,j] for j in range(len(dates))]) for i in range(len(nurses))]) + lpSum([w_iota[j]*iota[j] + e_3[j] for j in range(len(dates))]) + lpSum([w_lambda[i]*lamb[i]+add[i] for i in range(len(nurses))])  + lpSum([lpSum([w_u[i][i_dash]*u[i,i_dash] for i_dash in range(len(nurses))]) for i in range(len(nurses))])- lpSum([lpSum([lpSum([a[i,j,p] for p in range(0,4)]) for j in range(len(dates))]) for i in range(len(nurses))])

    #制約条件
    #(2)看護師はそれぞれ1つの勤務に割り当てられる
    for i in range(len(nurses)):
        for j in range(len(dates)):
            model += lpSum([x[i,j,k] for k in range(len(shifts))]) == 1

    #(3)日勤帯に必要な看護師の下限
    for j in range(len(dates)):
        model += lpSum([lpSum([x[i,j,k] for k in range(0,3)]) for i in range(len(nurses))]) >= 9 - iota[j]

    #(4)中勤帯に必要な看護師の下限
    for j in range(len(dates)):
        model += lpSum([lpSum([x[i,j,k] for k in range(2,4)]) for i in range(len(nurses))]) == 3 #- kappa[j]

    #(5)夜勤帯に必要な看護師の下限
    for j in range(len(dates)):
        model += lpSum([lpSum([x[i,j,k] for k in range(4,5)]) for i in range(len(nurses))]) == 3 #- alpha[j] + w_kappa[j]*kappa[j] + w_alpha[j]*alpha[j]

    #(6)月間総合勤務時間は145時間以上171時間以下
    for i in range(len(nurses)):
        #model += lpSum([lpSum([times[k]*x[i,j,k] for k in range(len(shifts))]) for j in range(len(dates))]) <= 171
        model += lpSum([lpSum([times[k]*x[i,j,k] for k in range(len(shifts))]) for j in range(len(dates))]) <= 145 + add[i]

    #(7)連続勤務は5日間まで
    for i in range(len(nurses)):
        for j in range(5,31):
            model += lpSum([lpSum([x[i,j-n,k] for k in range(len(shifts)-1)]) for n in range(0,6)]) <= 5

    #(8)夜勤は1か月6回まで 3回以上
    for i in range(len(nurses)):
        model += lpSum([x[i,j,4] for j in range(len(dates))]) <= 6

    #(9)連続夜勤は2日間まで
    for i in range(len(nurses)):
        for j in range(2,31):
            model += lpSum([x[i,j-n,4] for n in range(0,3)]) <= 2

    #クール1の成立 「夜・夜・休み・長・N4」
    for i in range(len(nurses)):
        for j in range(0,27):
            model += a[i,j,0] <= x[i,j,4]
            model += a[i,j,0] <= x[i,j+1,4]
            model += a[i,j,0] <= x[i,j+2,5]
            model += a[i,j,0] <= x[i,j+3,2]
            model += a[i,j,0] <= x[i,j+4,3]

    #クール2の成立 「夜・夜・休み・E3・長」
    for i in range(len(nurses)):
        for j in range(0,27):
            model += a[i,j,1] <= x[i,j,4]
            model += a[i,j,1] <= x[i,j+1,4]
            model += a[i,j,1] <= x[i,j+2,5]
            model += a[i,j,1] <= x[i,j+3,1]
            model += a[i,j,1] <= x[i,j+4,2]

    #クール3の成立 「夜・休み・長・N4⇒長」
    for i in range(len(nurses)):
        for j in range(0,28):
            model += a[i,j,2] <= x[i,j,4]
            model += a[i,j,2] <= x[i,j+1,5]
            model += a[i,j,2] <= x[i,j+2,2]
            model += a[i,j,2] <= x[i,j+3,3]+x[i,j+3,2]

    #クール4の成立 「夜・休み・E3・長⇒N4」
    for i in range(len(nurses)):
        for j in range(0,28):
            model += a[i,j,3] <= x[i,j,4]
            model += a[i,j,3] <= x[i,j+1,5]
            model += a[i,j,3] <= x[i,j+2,1]
            model += a[i,j,3] <= x[i,j+3,2]+x[i,j+3,3]

    #夜勤があったらどれかのクールになる
    for i in range(len(nurses)):
        model += x[i,j,4] <= a[i,0,0] + a[i,0,1] + a[i,0,2] + a[i,0,3]


    for i in range(len(nurses)):
        for j in range(1,27):
            model += x[i,j,4] <= a[i,j,0] + a[i,j-1,0] + a[i,j,1] + a[i,j-1,1] + a[i,j,2] + a[i,j,3]

    #クールの前日は休み
    for i in range(len(nurses)):
        for j in range(1,27):
            for p in range(0,2):
                model += a[i,j,p] <= x[i,j-1,5]

    for i in range(len(nurses)):
        for j in range(1,28):
            for p in range(2,4):
                model += a[i,j,p] <= x[i,j-1,5]

    #月末4日間で夜勤の前日は休み
    for i in range(len(nurses)):
        for j in range(27,31):
            model += x[i,j-1,5] >= x[i,j,4]

    #月末4日間で夜勤の次の日は休み
    for i in range(len(nurses)):
        for j in range(27,30):
            model += x[i,j+1,5] >= x[i,j,4]

    #(18),(19)土日休みが月1回以上
    for i in range(len(nurses)):
        for j in range(5,27,7):
            model += x[i,j,5] + x[i,j-1,5] >= 2*y[i,j]

    for i in range(len(nurses)):
        model += lpSum([y[i,j] for j in range(5,27,7)]) >= 1

    #(20)1日あたりE3が6人必要
    for j in range(len(dates)):
        model += lpSum([x[i,j,1] for i in range(len(nurses))]) >= 6 - e_3[j]

    #(21),(22)4日連続休みが月1回以上
    for i in range(len(nurses)):
        for j in range(3,31):
            model += lpSum([x[i,j-n,5] for n in range(0,4)]) >= 4*z[i,j]

    for i in range(len(nurses)):
        model += lpSum([z[i,j] for j in range(3,31)]) == 1 + lamb[i]

    #クール1,2に含まれる3,4はaを1にしない
    #for i in range(len(nurses)):
    #    for j in range(1,28):
    #        for p in range(2,4):
    #            model += a[i,j,p] + a[i,j-1,p-2] <= 1

    #(23),(24)クールの間は4日以上空く
    for i in range(len(nurses)):
        for j in range(0,21):
            for p in range(0,4):
                model += a[i,j,p] + lpSum([lpSum([a[i,t,q] for t in range(j+1,j+L[p]+5)]) for q in range(0,4)]) <= 1

    #(25)夜勤の回数はできるだけ公平に
    for i in range(len(nurses)):
        for i_dash in range(len(nurses)):
            model += lpSum([x[i,j,4]-x[i_dash,j,4] for j in range(len(dates))]) >= -1 * u[i,i_dash]

    for i in range(len(nurses)):
        for i_dash in range(len(nurses)):
            model += lpSum([x[i,j,4]-x[i_dash,j,4] for j in range(len(dates))]) <= u[i,i_dash]

    # 1回目のシフトと今回のシフトの差の絶対値はu_ijk以内
    for i in range(len(nurses)):
        for j in range(len(dates)):
            for k in range(len(shifts)):
                model += x[i,j,k] - schedule[i][j][k] <= h[i,j,k]
    for i in range(len(nurses)):
        for j in range(len(dates)):
            for k in range(len(shifts)):
                model += x[i,j,k] - schedule[i][j][k] >= (-1) * h[i,j,k]
    
    #(26)-(29)バイナリ制約、(30)-(41)非負制約 <= 変数の定義に含む

    #結果の出力
    solver = GUROBI_CMD(msg=True, timeLimit=60)  # 60秒で打ち切り
    status = model.solve(solver)

    #得た解を配列として保存
    result_shift = []
    for i in range(len(nurses)):
        j_content = []
        for j in range(len(dates)):
            k_content = []
            for k in range(len(shifts)):
                row = x[i,j,k].value()
                k_content.append(row)
            j_content.append(k_content)
        result_shift.append(j_content)
        
    return result_shift

def fit_fix_schedule(schedule):
    for i in range(len(nurses)):
        for j in range(len(dates)):
            max = schedule[i][j][0]
            for k in range(1,6):
                if schedule[i][j][k] > max:
                    max = schedule[i][j][k]
            for k in range(len(shifts)):
                if schedule[i][j][k] == max:
                    schedule[i][j][k] = 1
                else:
                    schedule[i][j][k] = 0
    return schedule

#def schedule_to_df_2(schedule):
#    df_matrix = []
#    for i in range(len(nurses)):
#        for j in range(len(dates)):
#            max = schedule[i][j][0]
#            for k in range(1,6):
#                if schedule[i][j][k] > max:
#                    max = schedule[i][j][k]
#            for k in range(len(shifts)):
#                if schedule[i][j][k] == max:
#                    schedule[i][j][k] = 1
#                else:
#                    schedule[i][j][k] = 0
#    fixed_schedule = fix_shift(schedule, nurses, dates, shifts)
#    for i in range(len(fixed_schedule)):
#        works = []
#        for j in range(len(fixed_schedule[0])):
#            shift = fixed_schedule[i][j]
#            idx = np.argmax(shift)
#            works.append(["日", "E3", "長", "N4", "夜", "休"][idx])
#        df_matrix.append(works)
#    for i in range(len(df_matrix)):
#        df_matrix[i].insert(0, nurses[i])
#    columns = ["名前"] + [f"{d}日" for d in dates]
#    return pd.DataFrame(df_matrix, columns=columns)

# === セッション初期化 ===
if "model" not in st.session_state:
    session_init()

# === UI ===
st.title("修正シフト逐次学習システム (Interactive Shift Learning System)")

sample_file = st.file_uploader("初期シフトCSVをアップロード（Upload Initial Shift CSV)", type="csv")
if sample_file:
    df = preprocess(sample_file)
    if st.session_state.current_schedule is None:
        st.session_state.current_schedule = df_to_schedule(df)

    st.subheader("編集可能な現在のシフト（Editable Current Shift Schedule）")
    df_display = schedule_to_df(st.session_state.current_schedule)
    column_config = {col: st.column_config.SelectboxColumn(label=col, options=category_options, required=True) for col in df_display.columns if col != "名前"}
    edited_df = st.data_editor(df_display, column_config=column_config, use_container_width=True, num_rows="dynamic", key="editor")

    if not st.session_state.learning_finished:
        if st.button("このシフトを学習する（Train on This Shift Schedule）"):
            edited_schedule = df_to_schedule(edited_df)
            st.session_state.edited_schedule = edited_schedule
            x_tensor = torch.tensor(st.session_state.current_schedule, dtype=torch.float32).unsqueeze(0)
            y_tensor = torch.tensor(edited_schedule, dtype=torch.float32).unsqueeze(0)
            new_schedule_tensor = train_step(x_tensor, y_tensor)
            st.session_state.current_schedule = new_schedule_tensor.squeeze(0).tolist()
            mse = calc_mse(st.session_state.current_schedule, edited_schedule)
            st.info(f"MSE: {mse:.4f}")
            st.session_state.current_schedule = fix_shift(fit_fix_schedule(new_schedule_tensor.squeeze(0).tolist()), nurses, dates, shifts)
            st.rerun()

    if st.button("学習を終了してモデルを確定する（Finish Learning and Finalize Model）"):
        st.session_state.learning_finished = True
        torch.save(st.session_state.model.state_dict(), "trained_model.pt")
        st.success("モデルを確定して保存しました。これ以降は学習されません。")

    if st.session_state.learning_finished:
        st.info("✅ このモデルは確定されています。修正・学習操作は無効です。")

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

    csv = convert_to_csv(schedule_to_df(st.session_state.current_schedule))
    st.download_button("このシフトをダウンロードする（Download This Shift Schedule）", csv, "修正から得られたシフト.csv", "text/csv", key="download-csv")

# === モデルテスト ===
st.markdown("---")
st.header("確定済みモデルのテスト")

model_file = st.file_uploader("確定モデル (.pt) をアップロード", type="pt")
test_file = st.file_uploader("テスト用シフトCSVをアップロード", type="csv")

# ✅ すでにテスト済みなら、再実行時にスキップする
if model_file and test_file and not st.session_state.get("testing_completed", False):
    st.session_state.testing_started = True
    st.session_state.testing_completed = True  # ✅ ここで「テスト完了」フラグを記録

    test_model = MyModel(input_dim, hidden_dim, output_dim, hidden2_dim, hidden3_dim)
    buffer = io.BytesIO(model_file.read())
    test_model.load_state_dict(torch.load(buffer))
    test_model.eval()

    test_df = preprocess(test_file)
    test_schedule = df_to_schedule(test_df)
    x_tensor = torch.tensor(test_schedule, dtype=torch.float32).unsqueeze(0).view(1, -1)

    with torch.no_grad():
        output_tensor = test_model(x_tensor)

    predicted_schedule = output_tensor.view(20, 31, 6).tolist()

    # ✅ current_schedule をテスト出力に更新
    st.session_state.current_schedule = predicted_schedule

    mse = calc_mse(test_schedule, predicted_schedule)
    st.write(f"MSE: {mse:.4f}")

    result_df = schedule_to_df(predicted_schedule)
    st.subheader("モデルによって生成されたシフト")
    st.dataframe(result_df)

    result_csv = convert_to_csv(result_df)
    st.download_button(
        "生成されたシフトをダウンロードする",
        result_csv,
        "テストで得られたシフト.csv",
        "text/csv",
        key="download-pred"
    )
