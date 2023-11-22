import gdown

# Đường link chia sẻ của file trên Google Drive
file_url = 'https://drive.google.com/drive/folders/14eDH4cCnQ81XS_s6b7FBfoVcm1Xnizlo'

# Tên file sau khi tải về
output_file = 'output_file_name.ext'

# Sử dụng hàm download từ thư viện gdown
gdown.download(file_url, output_file, quiet=False)
