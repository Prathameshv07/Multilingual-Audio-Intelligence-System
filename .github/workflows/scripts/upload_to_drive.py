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
import os
import json
import mimetypes
from google.oauth2.credentials import Credentials
from google.oauth2 import service_account
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload
import sys

# Target folder ID - This is where files will be uploaded
FOLDER_ID = '1-8HJcWxsUUQIj9OMXQeoeULS06RA9Hg9'

def create_oauth_service():
    """Try to create Google Drive service using OAuth credentials"""
    try:
        print("🔐 Attempting OAuth authentication...")
        
        # Load OAuth credentials from environment
        oauth_token_json = os.environ.get('GOOGLE_OAUTH_TOKEN')
        if not oauth_token_json:
            print("❌ GOOGLE_OAUTH_TOKEN not found in environment")
            return None
            
        token_info = json.loads(oauth_token_json)
        
        # Create credentials from the token info
        credentials = Credentials.from_authorized_user_info(token_info)
        
        # Refresh the token if needed
        if credentials.expired and credentials.refresh_token:
            print("🔄 Token expired, attempting refresh...")
            credentials.refresh(Request())
            print("✅ Token refreshed successfully")
        
        # Test the credentials by building service
        service = build('drive', 'v3', credentials=credentials)
        
        # Test access to the folder
        test_query = f"'{FOLDER_ID}' in parents and trashed=false"
        test_results = service.files().list(q=test_query, pageSize=1).execute()
        
        print("✅ OAuth authentication successful!")
        return service
        
    except Exception as e:
        print(f"❌ OAuth authentication failed: {str(e)}")
        return None

def create_service_account_service():
    """Try to create Google Drive service using Service Account credentials"""
    try:
        print("🔐 Attempting Service Account authentication...")
        
        # Load service account credentials from environment
        service_account_json = os.environ.get('GOOGLE_SERVICE_ACCOUNT_KEY')
        if not service_account_json:
            print("❌ GOOGLE_SERVICE_ACCOUNT_KEY not found in environment")
            return None
            
        credentials_info = json.loads(service_account_json)
        
        # Create credentials from service account
        credentials = service_account.Credentials.from_service_account_info(
            credentials_info,
            scopes=['https://www.googleapis.com/auth/drive.file']
        )
        
        # Build the Drive service
        service = build('drive', 'v3', credentials=credentials)
        
        # Test access to the folder
        test_query = f"'{FOLDER_ID}' in parents and trashed=false"
        test_results = service.files().list(q=test_query, pageSize=1).execute()
        
        print("✅ Service Account authentication successful!")
        return service
        
    except Exception as e:
        print(f"❌ Service Account authentication failed: {str(e)}")
        return None

def get_drive_service():
    """Get Google Drive service with fallback authentication"""
    print("🚀 Initializing Google Drive authentication with fallback...")
    
    # Try OAuth first
    service = create_oauth_service()
    if service:
        return service, "OAuth"
    
    print("🔄 OAuth failed, trying Service Account fallback...")
    
    # Fallback to Service Account
    service = create_service_account_service()
    if service:
        return service, "Service Account"
    
    # Both methods failed
    print("💥 Both authentication methods failed!")
    print("\nPlease ensure you have either:")
    print("1. GOOGLE_OAUTH_TOKEN secret set with valid OAuth credentials, OR")
    print("2. GOOGLE_SERVICE_ACCOUNT_KEY secret set with service account JSON")
    print("\nFor Service Account:")
    print("- Create a service account in Google Cloud Console")
    print("- Share your target folder with the service account email")
    print("- Add the service account JSON as GOOGLE_SERVICE_ACCOUNT_KEY secret")
    
    sys.exit(1)

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
        print(f'📝 Updated: {file_name} (ID: {updated_file.get("id")})')
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
        print(f'📤 Uploaded: {file_name} (ID: {file.get("id")})')

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
        print(f'📁 Created folder: {folder_name} (ID: {folder.get("id")})')
        return folder.get('id')

def upload_directory(local_path, parent_folder_id, drive_service, exclude_dirs=None, exclude_files=None):
    if exclude_dirs is None:
        exclude_dirs = ['.git', '.github', 'node_modules', '__pycache__']
    if exclude_files is None:
        exclude_files = ['*.md']  # Skip markdown files

    import fnmatch
    uploaded_count = 0
    skipped_count = 0
    error_count = 0

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
                print(f'⏭️  Skipping {file} (excluded file type)')
                skipped_count += 1
                continue

            file_path = os.path.join(root, file)
            try:
                upload_file(file_path, current_folder_id, drive_service)
                uploaded_count += 1
            except Exception as e:
                print(f'❌ Error uploading {file_path}: {e}')
                error_count += 1

    return uploaded_count, skipped_count, error_count

def main():
    """Main execution function"""
    print("=" * 60)
    print("🚀 ROBUST GOOGLE DRIVE UPLOADER WITH FALLBACK")
    print("=" * 60)
    
    # Get Drive service with fallback authentication
    service, auth_method = get_drive_service()
    
    print(f"🎉 Successfully authenticated using: {auth_method}")
    
    # Test folder permissions
    try:
        test_query = f"'{FOLDER_ID}' in parents and trashed=false"
        test_results = service.files().list(q=test_query, pageSize=1).execute()
        print(f"✅ Successfully accessed folder. Found {len(test_results.get('files', []))} items (showing 1 max)")
    except Exception as e:
        print(f"💥 ERROR: Cannot access folder {FOLDER_ID}")
        print(f"Error: {e}")
        if auth_method == "Service Account":
            print("💡 Make sure to share the folder with the service account email!")
        sys.exit(1)

    # Upload all files to Google Drive
    print("\n📤 Starting upload to Google Drive...")
    print("-" * 40)
    
    uploaded, skipped, errors = upload_directory('.', FOLDER_ID, service)
    
    print("-" * 40)
    print("📊 UPLOAD SUMMARY:")
    print(f"✅ Files uploaded: {uploaded}")
    print(f"⏭️  Files skipped: {skipped}")
    print(f"❌ Errors: {errors}")
    print(f"🔐 Authentication method: {auth_method}")
    print("=" * 60)
    
    if errors > 0:
        print("⚠️  Some files failed to upload. Check the logs above for details.")
        sys.exit(1)
    else:
        print("🎉 Upload completed successfully!")

if __name__ == "__main__":
    main()