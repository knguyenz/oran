import cvxpy as cp  # Import thư viện CVXPY để giải quyết bài toán tối ưu hóa
import numpy as np  # Import thư viện NumPy để tính toán mảng và số học
import pandas as pd  # Import thư viện pandas để làm việc với dữ liệu dạng bảng

# Load the CSV data
data = pd.read_csv('D:/test2/input.csv')  # Đọc dữ liệu từ file CSV (đường dẫn được chỉ định)

# Extract values from the CSV data
num_user = data['numuser'][0]  # Số lượng người dùng (numuser) từ dữ liệu
num_RU = data['numRU'][0]  # Số lượng Remote Unit (RU) từ dữ liệu
RBeachRU = eval(data['RBeachRU'][0])  # Chuyển chuỗi thành list các RBs per RU
Pmax = eval(data['Pmax'][0])  # Chuyển chuỗi thành list các giá trị Pmax cho từng RU
RminK = eval(data['RminK'][0])  # Chuyển chuỗi thành list các tốc độ dữ liệu tối thiểu (RminK) cho từng người dùng
Tmin = data['Tmin'][0]  # Tốc độ dịch vụ tối thiểu (Tmin)
BW = data['BW'][0]  # Băng thông của mỗi RB
N0 = data['N0'][0]  # Mật độ nhiễu (N0)

# Kiểm tra xem RBeachRU có phải là danh sách các danh sách không (mỗi danh sách là các RB của mỗi RU)
if isinstance(RBeachRU, list) and all(isinstance(i, list) for i in RBeachRU):
    num_RBs_per_RU = len(RBeachRU[0])  # Giả sử tất cả các RU có cùng số lượng RBs
else:
    num_RBs_per_RU = len(RBeachRU) if isinstance(RBeachRU, list) else RBeachRU  # Nếu không phải, lấy số lượng RBs

# Định nghĩa các biến của bài toán
xi_bk = {}  # Biến chỉ định RB của RU i được phân cho người dùng k
pi_bk = {}  # Biến công suất của RB b của RU i dành cho người dùng k
yik = {}  # Biến chỉ định RU i có phục vụ người dùng k không
pi_k = {}  # Biến chỉ định người dùng k có được phục vụ không
z_ibk = {}  # Đại diện cho thông lượng (throughput) của người dùng

# Khởi tạo các biến tối ưu hóa trong CVXPY
for i in range(num_RU):  # Duyệt qua tất cả các RU
    for b in range(num_RBs_per_RU):  # Duyệt qua tất cả các RBs của mỗi RU
        for k in range(num_user):  # Duyệt qua tất cả các người dùng
            xi_bk[(i, b, k)] = cp.Variable(boolean=True, name=f"xi_bk_{i}_{b}_{k}")  # Biến nhị phân xi_bk
            pi_bk[(i, b, k)] = cp.Variable(nonneg=True, name=f"pi_bk_{i}_{b}_{k}")  # Biến công suất pi_bk
            z_ibk[(i, b, k)] = cp.Variable(nonneg=True, name=f"z_ibk_{i}_{b}_{k}")  # Biến thông lượng z_ibk
    for k in range(num_user):  # Đảm bảo mỗi người dùng có một biến yik
        yik[(i, k)] = cp.Variable(boolean=True, name=f"yik_{i}_{k}")  # Biến nhị phân yik
    for k in range(num_user):  # Đảm bảo pi_k được khởi tạo cho mỗi người dùng
        pi_k[k] = cp.Variable(boolean=True, name=f"pi_k_{k}")  # Biến nhị phân pi_k

# Hàm mục tiêu (maximize throughput): Tối đa hóa tổng thông lượng cho tất cả các người dùng
objective = cp.Maximize(
    cp.sum([z_ibk[(i, b, k)] for i in range(num_RU) for b in range(num_RBs_per_RU) for k in range(num_user)])
)

# Các ràng buộc
constraints = []

# Liên kết giữa z_ibk và xi_bk, pi_bk
for i in range(num_RU):  # Duyệt qua tất cả các RU
    for b in range(num_RBs_per_RU):  # Duyệt qua tất cả các RBs của mỗi RU
        for k in range(num_user):  # Duyệt qua tất cả các người dùng
            constraints.append(z_ibk[(i, b, k)] <= pi_bk[(i, b, k)])  # z_ibk <= pi_bk
            constraints.append(z_ibk[(i, b, k)] <= Pmax[i] * xi_bk[(i, b, k)])  # z_ibk <= Pmax[i] * xi_bk
            constraints.append(z_ibk[(i, b, k)] >= pi_bk[(i, b, k)] - (1 - xi_bk[(i, b, k)]) * Pmax[i])  # Công suất z_ibk >= giá trị công suất từ xi_bk và pi_bk

# Ràng buộc 1: Mỗi RB chỉ có thể được gán tối đa 1 user
for i in range(num_RU):  # Duyệt qua tất cả các RU
    for b in range(num_RBs_per_RU):  # Duyệt qua tất cả các RBs
        constraints.append(cp.sum([xi_bk[(i, b, k)] for k in range(num_user)]) <= 1)  # Ràng buộc rằng mỗi RB chỉ có thể được gán cho một người dùng

# Ràng buộc 2: Đảm bảo tốc độ dữ liệu tối thiểu cho mỗi người dùng (tuyến tính hóa hàm log)
for k in range(num_user):  # Duyệt qua tất cả các người dùng
    data_rate = cp.sum([  # Tính tổng tốc độ dữ liệu
        BW * cp.log(1 + ((z_ibk[(i, b, k)] * RBeachRU[i]) / (BW * N0)))  # Hàm log2 tuyến tính
        for i in range(num_RU) for b in range(num_RBs_per_RU)
    ])
    constraints.append(data_rate >= RminK[k] * pi_k[k])  # Ràng buộc rằng tốc độ dữ liệu phải lớn hơn hoặc bằng tốc độ tối thiểu của người dùng k

# Ràng buộc 3: Mối quan hệ giữa xi và y
for k in range(num_user):  # Duyệt qua tất cả các người dùng
    for i in range(num_RU):  # Duyệt qua tất cả các RU
        lhs = cp.sum([xi_bk[(i, b, k)] for b in range(num_RBs_per_RU)]) / num_RBs_per_RU  # Tính tổng xi_bk trên các RB của RU
        constraints.append(lhs <= yik[(i, k)] + 1e-5)  # Ràng buộc giữa xi_bk và yik

# Ràng buộc 4: Mối quan hệ giữa pi và y
for k in range(num_user):  # Duyệt qua tất cả các người dùng
    lhs = cp.sum([yik[(i, k)] for i in range(num_RU)]) / num_RU  # Tính tổng yik cho tất cả các RU
    constraints.append(lhs <= pi_k[k] + 1e-5)  # Ràng buộc giữa pi_k và yik

# Ràng buộc 5: Tổng công suất truyền không vượt quá Pmax của RU
for i in range(num_RU):  # Duyệt qua tất cả các RU
    total_power = cp.sum([z_ibk[(i, b, k)] for b in range(num_RBs_per_RU) for k in range(num_user)])  # Tổng công suất truyền
    constraints.append(total_power <= Pmax[i] + 1e-5)  # Ràng buộc rằng tổng công suất không vượt quá Pmax của RU

# Ràng buộc 6: Quan hệ giữa u, p và x (đã được chuyển đổi)
for k in range(num_user):  # Duyệt qua tất cả các người dùng
    for i in range(num_RU):  # Duyệt qua tất cả các RU
        for b in range(num_RBs_per_RU):  # Duyệt qua tất cả các RBs
            constraints.append(z_ibk[(i, b, k)] <= pi_bk[(i, b, k)])  # z_ibk <= pi_bk
            constraints.append(z_ibk[(i, b, k)] <= Pmax[i] * xi_bk[(i, b, k)])  # z_ibk <= Pmax[i] * xi_bk
            constraints.append(z_ibk[(i, b, k)] >= pi_bk[(i, b, k)] - Pmax[i] * xi_bk[(i, b, k)])  # Quan hệ giữa z_ibk và xi_bk, pi_bk

# Giải bài toán tối ưu
problem = cp.Problem(objective, constraints)  # Tạo bài toán tối ưu trong CVXPY
problem.solve(solver=cp.MOSEK)  # Giải bài toán với solver MOSEK

# Lưu kết quả vào file output.txt
output_data = {  # Lưu giá trị của các biến vào dictionary
    "pi_bk": {key: var.value for key, var in pi_bk.items()},
    "z_ibk": {key: var.value for key, var in z_ibk.items()},
    "yik": {key: var.value for key, var in yik.items()},
    "pi_k": {key: var.value for key, var in pi_k.items()},
}

output_file_path = 'output.txt'  # Định nghĩa đường dẫn file đầu ra
with open(output_file_path, 'w') as file:  # Mở file để ghi kết quả
    for var_name, values in output_data.items():  # Duyệt qua tất cả các biến và giá trị của chúng
        for key, value in values.items():
            file.write(f"{key}: {value}\n")  # Ghi giá trị vào file

print(f"Optimization results saved to {output_file_path}")  # Thông báo kết quả đã được lưu vào file
