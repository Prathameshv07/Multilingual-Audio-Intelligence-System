#!/usr/bin/env python3
import os
import json
import mimetypes
import fnmatch
import sys
from google.oauth2.credentials import Credentials
from google.oauth2 import service_account
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload

FOLDER_ID = '1-8HJcWxsUUQIj9OMXQeoeULS06RA9Hg9'

def get_drive_service():
    """Get Google Drive service with OAuth/Service Account fallback"""
    # Try OAuth first
    oauth_token = os.environ.get('GOOGLE_OAUTH_TOKEN')
    if oauth_token:
        try:
            token_info = json.loads(oauth_token)
            credentials = Credentials.from_authorized_user_info(token_info)
            if credentials.expired and credentials.refresh_token:
                credentials.refresh(Request())
            return build('drive', 'v3', credentials=credentials)
        except Exception as e:
            print(f"OAuth failed: {e}")
    
    # Fallback to Service Account
    sa_key = os.environ.get('GOOGLE_SERVICE_ACCOUNT_KEY')
    if sa_key:
        try:
            credentials_info = json.loads(sa_key)
            credentials = service_account.Credentials.from_service_account_info(
                credentials_info, scopes=['https://www.googleapis.com/auth/drive.file']
            )
            return build('drive', 'v3', credentials=credentials)
        except Exception as e:
            print(f"Service Account failed: {e}")
    
    print("Both authentication methods failed!")
    sys.exit(1)

def upload_file(file_path, parent_folder_id, service):
    """Upload or update a file"""
    file_name = os.path.basename(file_path)
    
    # Check if file exists
    query = f"name='{file_name}' and '{parent_folder_id}' in parents and trashed=false"
    results = service.files().list(q=query).execute()
    items = results.get('files', [])
    
    media = MediaFileUpload(file_path, mimetype=mimetypes.guess_type(file_path)[0] or 'application/octet-stream')
    
    if items:
        # Update existing
        service.files().update(fileId=items[0]['id'], media_body=media).execute()
        print(f'Updated: {file_name}')
    else:
        # Create new
        file_metadata = {'name': file_name, 'parents': [parent_folder_id]}
        service.files().create(body=file_metadata, media_body=media).execute()
        print(f'Uploaded: {file_name}')

def create_folder(folder_name, parent_id, service):
    """Create folder if it doesn't exist"""
    query = f"name='{folder_name}' and '{parent_id}' in parents and mimeType='application/vnd.google-apps.folder' and trashed=false"
    results = service.files().list(q=query).execute()
    items = results.get('files', [])
    
    if items:
        return items[0]['id']
    
    folder_metadata = {
        'name': folder_name,
        'parents': [parent_id],
        'mimeType': 'application/vnd.google-apps.folder'
    }
    folder = service.files().create(body=folder_metadata).execute()
    return folder.get('id')

def upload_directory(local_path, parent_id, service):
    """Upload directory with exclusions"""
    exclude_dirs = {'.git', '.github', 'node_modules', '__pycache__'}
    exclude_patterns = ['*.md']  # Skip markdown files
    
    uploaded = skipped = errors = 0
    
    for root, dirs, files in os.walk(local_path):
        # Filter directories
        dirs[:] = [d for d in dirs if d not in exclude_dirs]
        
        # Create folder structure
        rel_path = os.path.relpath(root, local_path)
        current_folder_id = parent_id
        
        if rel_path != '.':
            for part in rel_path.split(os.sep):
                current_folder_id = create_folder(part, current_folder_id, service)
        
        # Upload files
        for file in files:
            # Check exclusions
            if any(fnmatch.fnmatch(file, pattern) for pattern in exclude_patterns):
                print(f'Skipping: {file}')
                skipped += 1
                continue
            
            try:
                upload_file(os.path.join(root, file), current_folder_id, service)
                uploaded += 1
            except Exception as e:
                print(f'Error uploading {file}: {e}')
                errors += 1
    
    return uploaded, skipped, errors

def main():
    print("Starting Google Drive upload...")
    
    # Get service and test access
    service = get_drive_service()
    
    try:
        service.files().list(q=f"'{FOLDER_ID}' in parents", pageSize=1).execute()
        print("Folder access confirmed")
    except Exception as e:
        print(f"Cannot access folder {FOLDER_ID}: {e}")
        sys.exit(1)
    
    # Upload files
    uploaded, skipped, errors = upload_directory('.', FOLDER_ID, service)
    
    print(f"\nSummary: {uploaded} uploaded, {skipped} skipped, {errors} errors")
    
    if errors > 0:
        sys.exit(1)
    print("Upload completed successfully!")

if __name__ == "__main__":
    main()