import os
import json
import mimetypes
from google.oauth2.credentials import Credentials
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload

# Load OAuth credentials from environment
oauth_token_json = os.environ['GOOGLE_OAUTH_TOKEN']
token_info = json.loads(oauth_token_json)

# Create credentials from the token info
credentials = Credentials.from_authorized_user_info(token_info)

# Refresh the token if needed
if credentials.expired and credentials.refresh_token:
    credentials.refresh(Request())

# Build the Drive service
service = build('drive', 'v3', credentials=credentials)

# Target folder ID - This is where files will be uploaded
FOLDER_ID = '1-8HJcWxsUUQIj9OMXQeoeULS06RA9Hg9'

def get_mime_type(file_path):
    mime_type, _ = mimetypes.guess_type(file_path)
    return mime_type or 'application/octet-stream'

def upload_file(file_path, parent_folder_id, drive_service):
    file_name = os.path.basename(file_path)

    # Check if file already exists in the specific folder
    query = f"name='{file_name}' and '{parent_folder_id}' in parents and trashed=false"
    results = drive_service.files().list(q=query).execute()
    items = results.get('files', [])

    media = MediaFileUpload(file_path, mimetype=get_mime_type(file_path), resumable=True)

    if items:
        # Update existing file
        file_id = items[0]['id']
        updated_file = drive_service.files().update(
            fileId=file_id,
            media_body=media
        ).execute()
        print(f'Updated: {file_name} (ID: {updated_file.get("id")})')
    else:
        # Create new file
        file_metadata = {
            'name': file_name,
            'parents': [parent_folder_id]
        }
        file = drive_service.files().create(
            body=file_metadata,
            media_body=media,
            fields='id'
        ).execute()
        print(f'Uploaded: {file_name} (ID: {file.get("id")})')

def create_folder_if_not_exists(folder_name, parent_folder_id, drive_service):
    """Create a folder if it doesn't exist and return its ID"""
    query = (
        f"name='{folder_name}' and '{parent_folder_id}' in parents and "
        f"mimeType='application/vnd.google-apps.folder' and trashed=false"
    )
    results = drive_service.files().list(q=query).execute()
    items = results.get('files', [])

    if items:
        return items[0]['id']
    else:
        folder_metadata = {
            'name': folder_name,
            'parents': [parent_folder_id],
            'mimeType': 'application/vnd.google-apps.folder'
        }
        folder = drive_service.files().create(body=folder_metadata, fields='id').execute()
        print(f'Created folder: {folder_name} (ID: {folder.get("id")})')
        return folder.get('id')

def upload_directory(local_path, parent_folder_id, drive_service, exclude_dirs=None, exclude_files=None):
    if exclude_dirs is None:
        exclude_dirs = ['.git', '.github', 'node_modules', '__pycache__']
    if exclude_files is None:
        exclude_files = ['*.md']  # Skip markdown files

    import fnmatch

    for root, dirs, files in os.walk(local_path):
        # Remove excluded directories
        dirs[:] = [d for d in dirs if d not in exclude_dirs]

        # Calculate relative path from the root
        rel_path = os.path.relpath(root, local_path)
        current_folder_id = parent_folder_id

        # Create nested folders if needed
        if rel_path != '.':
            path_parts = rel_path.split(os.sep)
            for part in path_parts:
                current_folder_id = create_folder_if_not_exists(part, current_folder_id, drive_service)

        # Upload files in current directory
        for file in files:
            should_skip = False
            for pattern in exclude_files:
                if fnmatch.fnmatch(file, pattern):
                    should_skip = True
                    break

            if should_skip:
                print(f'Skipping {file} (excluded file type)')
                continue

            file_path = os.path.join(root, file)
            try:
                upload_file(file_path, current_folder_id, drive_service)
            except Exception as e:
                print(f'Error uploading {file_path}: {e}')

# Test folder permissions first
try:
    test_query = f"'{FOLDER_ID}' in parents and trashed=false"
    test_results = service.files().list(q=test_query, pageSize=1).execute()
    print(f"Successfully accessed folder. Found {len(test_results.get('files', []))} items (showing 1 max)")
except Exception as e:
    print(f"ERROR: Cannot access folder {FOLDER_ID}. Error: {e}")
    exit(1)

# Upload all files to Google Drive (excluding MD files)
print("Starting upload to Google Drive...")
upload_directory('.', FOLDER_ID, service)

print("Upload completed - MD files were skipped, PDFs were uploaded!")
