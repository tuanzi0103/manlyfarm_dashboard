from pydrive2.auth import GoogleAuth
from pydrive2.drive import GoogleDrive

def upload_file(local_path, title):
    gauth = GoogleAuth()
    gauth.LocalWebserverAuth()   # 第一次会打开浏览器授权
    drive = GoogleDrive(gauth)

    f = drive.CreateFile({'title': title})
    f.SetContentFile(local_path)
    f.Upload()
    print(f"✅ Uploaded {title} to Google Drive")
    print(f"   File ID: {f['id']}")
    return f['id']

if __name__ == "__main__":
    ids = {}
    ids["transactions.csv"] = upload_file("transactions.csv", "transactions.csv")
    ids["inventory.csv"] = upload_file("inventory.csv", "inventory.csv")
    ids["members.csv"] = upload_file("members.csv", "members.csv")
    print("\n📌 Paste these IDs into app.py FILE_ID_MAP:")
    print(ids)
