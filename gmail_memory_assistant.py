#!/usr/bin/env python3
"""
Gmail Memory Assistant - Interactive script to fetch emails and create memories.

This script:
1. Prompts user for Gmail credentials (app password or OAuth)
2. Connects to Gmail API
3. Browses/fetches recent emails
4. Extracts key information to create "memories"
5. Stores memories in the souvenir_assistant database

Usage:
    python gmail_memory_assistant.py
"""

import os
import sys
import json
import pickle
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()

from google.oauth2.credentials import Credentials
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from google_auth_oauthlib.flow import InstalledAppFlow

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from souvenir_assistant import SouvenirAssistant


# Gmail API scopes
SCOPES = [
    'https://www.googleapis.com/auth/gmail.readonly',
    'https://www.googleapis.com/auth/gmail.labels'
]

TOKEN_FILE = 'gmail_token.pickle'
CREDS_FILE = 'gmail_credentials.json'


class GmailMemoryAssistant:
    """Gmail-powered memory assistant using souvenir storage."""
    
    def __init__(self, souvenir_db: str = "souvenir_memory.sqlite"):
        """Initialize Gmail memory assistant."""
        self.service = None
        self.credentials = None
        self.souvenir_assistant = SouvenirAssistant(db_path=souvenir_db)
        self.user_email = None
    
    def get_credentials_interactive(self) -> Optional[Credentials]:
        """
        Prompt user for Gmail credentials interactively.
        Supports both OAuth and app password authentication.
        Checks for GMAIL_ID and GMAIL_PASSWORD environment variables first.
        """
        # Check for environment variables first
        OAUTH_id = os.environ.get('GMAIL_OAUTH_CLIENT_ID')
        OAUTH_password = os.environ.get('GMAIL_OAUTH_CLIENT_SECRET')
        
        if OAUTH_id and OAUTH_password:
            print("\n" + "="*60)
            print("📧 Gmail Authentication")
            print("="*60)
            print("\n✅ Using credentials from environment variables")
            return self._authenticate_oauth()
        
        print("\n" + "="*60)
        print("📧 Gmail Authentication")
        print("="*60)
        print("\nChoose authentication method:")
        print("  1. OAuth 2.0 (recommended - uses browser)")
        print("  2. App Password (for 2FA accounts)")
        print("  3. Use existing token file")
        print("  4. Skip authentication (demo mode)")
        
        choice = input("\nEnter choice (1-4): ").strip()
        
        if choice == "1":
            return self._authenticate_oauth()
        elif choice == "2":
            return self._authenticate_app_password()
        elif choice == "3":
            return self._load_existing_token()
        elif choice == "4":
            print("\n⚠️  Running in demo mode - limited functionality")
            return None
        else:
            print("Invalid choice. Running in demo mode.")
            return None
    
    def _authenticate_oauth(self) -> Optional[Credentials]:
        """Authenticate using OAuth 2.0 flow."""
        print("\n🔐 OAuth 2.0 Authentication")
        print("-" * 40)
        
        # Check for OAuth credentials in environment variables first
        client_id = os.environ.get('GMAIL_OAUTH_CLIENT_ID')
        client_secret = os.environ.get('GMAIL_OAUTH_CLIENT_SECRET')
        
        if client_id and client_secret:
            print("✅ Using OAuth credentials from environment variables")
        else:
            # Prompt for OAuth credentials if not in environment
            client_id = input("Enter OAuth Client ID: ").strip()
            client_secret = input("Enter OAuth Client Secret: ").strip()
        
        if not client_id or not client_secret:
            print("❌ Client ID and Secret are required")
            return None
        
        # Save to temp file for the flow
        creds_data = {
            "client_id": client_id,
            "client_secret": client_secret
        }
        
        try:
            
            # Create a minimal client config for the flow
            client_config = {
                'installed': {
                    'client_id': client_id,
                    'client_secret': client_secret,
                    'redirect_uris': ['http://localhost'],
                    'auth_uri': 'https://accounts.google.com/o/oauth2/auth',
                    'token_uri': 'https://oauth2.googleapis.com/token'
                }
            }
            
            flow = InstalledAppFlow.from_client_config(
                client_config, 
                SCOPES
            )
            
            print("\n🌐 Opening browser for authentication...")
            print("(If no browser opens, copy the URL shown below)")
            
            # Run local server for callback
            self.credentials = flow.run_local_server(
                host="127.0.0.1",
                port=8000, 
                prompt='consent'
            )
            
            # Save token
            self._save_token()
            
            print("✅ OAuth authentication successful!")
            return self.credentials
            
        except Exception as e:
            print(f"❌ OAuth authentication failed: {e}")
            return None
    
    def _authenticate_app_password(self) -> Optional[Credentials]:
        """Authenticate using app password (for 2FA accounts)."""
        print("\n🔑 App Password Authentication")
        print("-" * 40)
        
        email = input("Enter Gmail address: ").strip()
        app_password = input("Enter App Password: ").strip()
        
        if not email or not app_password:
            print("❌ Email and App Password are required")
            return None
        
        # Validate email format
        if '@gmail.com' not in email and '@googlemail.com' not in email:
            print("⚠️  Warning: This doesn't look like a Gmail address")
            confirm = input("Continue anyway? (y/n): ").strip().lower()
            if confirm != 'y':
                return None
        
        # Create credentials using OAuth2 with refresh token flow
        # Note: This is a workaround - app passwords require OAuth2 token refresh
        try:
            from google.oauth2.credentials import Credentials
            
            # For app passwords, we need to create a token manually
            # This is a simplified approach
            self.credentials = Credentials(
                token=None,  # No access token yet
                refresh_token=None,
                token_uri='https://oauth2.googleapis.com/token',
                client_id='desktop_client',  # Dummy for app password
                client_secret=app_password,
                scopes=SCOPES
            )
            
            # Store email for later use
            self.user_email = email
            
            # Try to build service to verify
            try:
                service = build('gmail', 'v1', credentials=self.credentials)
                # Verify by getting profile
                profile = service.users().getProfile(userId='me').execute()
                self.user_email = profile.get('emailAddress', email)
                print(f"✅ Authenticated as: {self.user_email}")
                return self.credentials
            except HttpError as e:
                if e.resp.status == 401:
                    print("❌ Invalid app password. Please check and try again.")
                else:
                    print(f"❌ Authentication error: {e}")
                return None
                
        except Exception as e:
            print(f"❌ App password authentication failed: {e}")
            return None
    
    def _authenticate_app_password_env(self, email: str, app_password: str) -> Optional[Credentials]:
        """Authenticate using app password from environment variables."""
        print("\n🔑 App Password Authentication (from environment)")
        print("-" * 40)
        
        if not email or not app_password:
            print("❌ Email and App Password are required")
            return None
        
        # Validate email format
        if '@gmail.com' not in email and '@googlemail.com' not in email:
            print("⚠️  Warning: This doesn't look like a Gmail address")
        
        # Create credentials using OAuth2 with refresh token flow
        try:
            from google.oauth2.credentials import Credentials
            
            # For app passwords, we need to create a token manually
            self.credentials = Credentials(
                token=None,  # No access token yet
                refresh_token=None,
                token_uri='https://oauth2.googleapis.com/token',
                client_id='desktop_client',  # Dummy for app password
                client_secret=app_password,
                scopes=SCOPES
            )
            
            # Store email for later use
            self.user_email = email
            
            # Try to build service to verify
            try:
                service = build('gmail', 'v1', credentials=self.credentials)
                # Verify by getting profile
                profile = service.users().getProfile(userId='me').execute()
                self.user_email = profile.get('emailAddress', email)
                print(f"✅ Authenticated as: {self.user_email}")
                return self.credentials
            except HttpError as e:
                if e.resp.status == 401:
                    print("❌ Invalid app password. Please check and try again.")
                else:
                    print(f"❌ Authentication error: {e}")
                return None
                
        except Exception as e:
            print(f"❌ App password authentication failed: {e}")
            return None
    
    def _load_existing_token(self) -> Optional[Credentials]:
        """Load existing OAuth token from file."""
        if not os.path.exists(TOKEN_FILE):
            print("❌ No existing token file found")
            return None
        
        try:
            with open(TOKEN_FILE, 'rb') as f:
                self.credentials = pickle.load(f)
            
            # Check if token is valid
            if self.credentials and self.credentials.expired and self.credentials.refresh_token:
                print("🔄 Refreshing expired token...")
                self.credentials.refresh(Request())
                self._save_token()
            
            # Verify
            service = build('gmail', 'v1', credentials=self.credentials)
            profile = service.users().getProfile(userId='me').execute()
            self.user_email = profile.get('emailAddress')
            print(f"✅ Loaded existing session: {self.user_email}")
            return self.credentials
            
        except Exception as e:
            print(f"❌ Failed to load token: {e}")
            return None
    
    def _save_token(self):
        """Save credentials to token file."""
        if self.credentials:
            try:
                with open(TOKEN_FILE, 'wb') as f:
                    pickle.dump(self.credentials, f)
                print("💾 Token saved for future use")
            except Exception as e:
                print(f"⚠️  Could not save token: {e}")
    
    def connect(self, credentials: Optional[Credentials] = None) -> bool:
        """Connect to Gmail API."""
        if credentials:
            self.credentials = credentials
        elif not self.credentials:
            print("❌ No credentials available")
            return False
        
        try:
            self.service = build('gmail', 'v1', credentials=self.credentials)
            
            # Get user info
            if not self.user_email:
                profile = self.service.users().getProfile(userId='me').execute()
                self.user_email = profile.get('emailAddress')
            
            print(f"✅ Connected to Gmail as: {self.user_email}")
            return True
            
        except Exception as e:
            print(f"❌ Failed to connect to Gmail: {e}")
            return False
    
    def fetch_recent_emails(self, max_results: int = 50, days_back: int = 30) -> List[Dict]:
        """Fetch recent emails from Gmail."""
        if not self.service:
            print("❌ Not connected to Gmail")
            return []
        
        try:
            # Calculate date filter
            date_since = (datetime.now() - timedelta(days=days_back)).strftime('%Y/%m/%d')
            
            print(f"\n📬 Fetching emails from last {days_back} days...")
            
            # Get messages
            results = self.service.users().messages().list(
                userId='me',
                maxResults=max_results,
                q=f'after:{date_since}'
            ).execute()
            
            messages = results.get('messages', [])
            
            if not messages:
                print("📭 No recent emails found")
                return []
            
            print(f"📧 Found {len(messages)} emails. Fetching details...")
            
            emails = []
            for i, msg in enumerate(messages):
                try:
                    message = self.service.users().messages().get(
                        userId='me',
                        id=msg['id'],
                        format='full'
                    ).execute()
                    
                    # Extract key info
                    email_data = self._extract_email_info(message)
                    emails.append(email_data)
                    
                    if (i + 1) % 10 == 0:
                        print(f"  Processed {i + 1}/{len(messages)} emails...")
                        
                except HttpError as e:
                    print(f"⚠️  Error fetching email {msg['id']}: {e}")
                    continue
            
            print(f"✅ Successfully processed {len(emails)} emails")
            return emails
            
        except HttpError as e:
            print(f"❌ Error fetching emails: {e}")
            return []
    
    def _extract_email_info(self, message: Dict) -> Dict:
        """Extract relevant information from email message."""
        headers = message.get('payload', {}).get('headers', {})
        
        # Create header lookup
        header_dict = {h['name'].lower(): h['value'] for h in headers}
        
        # Get subject and sender
        subject = header_dict.get('subject', '(No Subject)')
        sender = header_dict.get('from', 'Unknown')
        to = header_dict.get('to', '')
        date = header_dict.get('date', '')
        
        # Get email body
        body = self._extract_body(message.get('payload', {}))
        
        # Get labels/categories
        labels = message.get('labelIds', [])
        
        return {
            'id': message['id'],
            'subject': subject,
            'sender': sender,
            'to': to,
            'date': date,
            'body': body,
            'labels': labels,
            'snippet': message.get('snippet', '')
        }
    
    def _extract_body(self, payload: Dict) -> str:
        """Extract email body from payload."""
        # Try to get body from parts
        if 'parts' in payload:
            for part in payload['parts']:
                if part.get('mimeType') == 'text/plain':
                    if 'data' in part.get('body', {}):
                        import base64
                        return base64.urlsafe_b64decode(
                            part['body']['data']
                        ).decode('utf-8', errors='ignore')
        
        # Try direct body
        if 'body' in payload and 'data' in payload['body']:
            import base64
            return base64.urlsafe_b64decode(
                payload['body']['data']
            ).decode('utf-8', errors='ignore')
        
        return ''
    
    def create_memories_from_emails(self, emails: List[Dict]) -> Dict[str, Any]:
        """Create souvenirs from extracted email data."""
        if not emails:
            return {"ok": False, "error": "No emails to process"}
        
        print(f"\n📝 Creating memories from {len(emails)} emails...")
        
        memories_created = 0
        categories_map = {
            'INBOX': 'email_inbox',
            'SENT': 'email_sent',
            'DRAFT': 'email_draft',
            'SPAM': 'email_spam',
            'TRASH': 'email_trash',
            'STARRED': 'email_starred',
            'IMPORTANT': 'email_important',
            'CATEGORY_PERSONAL': 'email_personal',
            'CATEGORY_WORK': 'email_work',
            'CATEGORY_SOCIAL': 'email_social',
            'CATEGORY_PROMOTIONS': 'email_promotions',
            'CATEGORY_UPDATES': 'email_updates',
        }
        
        for email in emails:
            # Determine category based on labels
            category = 'email_general'
            for label in email.get('labels', []):
                if label in categories_map:
                    category = categories_map[label]
                    break
            
            # Extract key info for memory content
            subject = email.get('subject', '')
            sender = email.get('sender', '')
            date = email.get('date', '')
            snippet = email.get('snippet', '')
            
            # Create a concise memory content
            memory_content = self._create_memory_content(email)
            
            # Extract tags from subject and sender
            tags = self._extract_tags(email)
            
            # Add to souvenir assistant
            result = self.souvenir_assistant.add_souvenir(
                content=memory_content,
                title=f"Email: {subject[:50]}{'...' if len(subject) > 50 else ''}",
                category=category,
                tags=tags
            )
            
            if result.get('ok'):
                memories_created += 1
        
        return {
            "ok": True,
            "memories_created": memories_created,
            "emails_processed": len(emails),
            "message": f"Created {memories_created} memories from {len(emails)} emails"
        }
    
    def _create_memory_content(self, email: Dict) -> str:
        """Create formatted memory content from email."""
        subject = email.get('subject', '')
        sender = email.get('sender', '')
        date = email.get('date', '')
        snippet = email.get('snippet', '')
        
        # Create a summary
        content = f"Email from {sender}"
        if date:
            content += f" on {date}"
        content += f"\n\nSubject: {subject}"
        
        if snippet:
            content += f"\n\nPreview: {snippet}"
        
        return content
    
    def _extract_tags(self, email: Dict) -> List[str]:
        """Extract relevant tags from email."""
        tags = []
        
        # Add sender domain as tag
        sender = email.get('sender', '')
        if '<' in sender:
            email_addr = sender.split('<')[1].rstrip('>')
        else:
            email_addr = sender
        
        if '@' in email_addr:
            domain = email_addr.split('@')[1].split('.')[0]
            tags.append(domain)
        
        # Add label-based tags
        for label in email.get('labels', []):
            if label.startswith('CATEGORY_'):
                tags.append(label.replace('CATEGORY_', '').lower())
        
        # Add subject keywords as tags (first 3 words)
        subject = email.get('subject', '').lower()
        words = [w for w in subject.split() if len(w) > 3][:3]
        tags.extend(words)
        
        return list(set(tags))[:10]  # Limit to 10 tags
    
    def browse_emails_interactive(self) -> List[Dict]:
        """Browse and select emails interactively."""
        if not self.service:
            print("❌ Not connected to Gmail")
            return []
        
        print("\n" + "="*60)
        print("📬 Email Browser")
        print("="*60)
        
        print("\nSearch options:")
        print("  1. All recent emails")
        print("  2. Search by query")
        print("  3. Search by sender")
        print("  4. Search by subject keyword")
        
        choice = input("\nEnter choice (1-4): ").strip()
        
        max_results = input("Max results (default 20): ").strip()
        max_results = int(max_results) if max_results.isdigit() else 20
        
        query = ""
        
        if choice == "1":
            # All recent
            date_since = (datetime.now() - timedelta(days=30)).strftime('%Y/%m/%d')
            query = f'after:{date_since}'
        elif choice == "2":
            query = input("Enter search query: ").strip()
        elif choice == "3":
            sender = input("Enter sender email: ").strip()
            query = f'from:{sender}'
        elif choice == "4":
            keyword = input("Enter subject keyword: ").strip()
            query = f'subject:{keyword}'
        else:
            print("Invalid choice")
            return []
        
        try:
            print(f"\n🔍 Searching with: {query}")
            
            results = self.service.users().messages().list(
                userId='me',
                maxResults=max_results,
                q=query
            ).execute()
            
            messages = results.get('messages', [])
            
            if not messages:
                print("📭 No emails found")
                return []
            
            print(f"\n📧 Found {len(messages)} emails:\n")
            
            # Show email list
            emails = []
            for i, msg in enumerate(messages[:20]):  # Show max 20
                try:
                    message = self.service.users().messages().get(
                        userId='me',
                        id=msg['id'],
                        format='metadata',
                        metadataHeaders=['From', 'Subject', 'Date']
                    ).execute()
                    
                    headers = {h['name'].lower(): h['value'] 
                              for h in message.get('payload', {}).get('headers', [])}
                    
                    print(f"{i+1}. {headers.get('subject', '(No Subject)')[:60]}")
                    print(f"   From: {headers.get('from', 'Unknown')[:50]}")
                    print(f"   Date: {headers.get('date', '')[:30]}")
                    print()
                    
                    # Store for selection
                    emails.append({
                        'id': msg['id'],
                        'subject': headers.get('subject', ''),
                        'sender': headers.get('from', ''),
                        'date': headers.get('date', '')
                    })
                    
                except Exception as e:
                    print(f"⚠️  Error: {e}")
                    continue
            
            # Let user select emails to save
            selection = input("Enter numbers to save as memories (comma-separated, or 'all'): ").strip()
            
            selected = []
            if selection.lower() == 'all':
                selected = emails
            else:
                for num in selection.split(','):
                    try:
                        idx = int(num.strip()) - 1
                        if 0 <= idx < len(emails):
                            selected.append(emails[idx])
                    except ValueError:
                        continue
            
            # Fetch full details for selected emails
            print(f"\n📥 Fetching {len(selected)} emails...")
            selected_emails = []
            for msg_id in [e['id'] for e in selected]:
                try:
                    message = self.service.users().messages().get(
                        userId='me',
                        id=msg_id,
                        format='full'
                    ).execute()
                    selected_emails.append(self._extract_email_info(message))
                except Exception as e:
                    print(f"⚠️  Error: {e}")
            
            return selected_emails
            
        except HttpError as e:
            print(f"❌ Search error: {e}")
            return []
    
    def run_interactive(self):
        """Run the interactive Gmail memory assistant."""
        print("\n" + "="*60)
        print("🎒 Gmail Memory Assistant")
        print("="*60)
        print("\nThis assistant will:")
        print("  1. Connect to your Gmail account")
        print("  2. Browse and select emails")
        print("  3. Create memories from selected emails")
        print("  4. Store them for future questions")
        
        # Get credentials
        self.credentials = self.get_credentials_interactive()
        
        if self.credentials:
            if not self.connect():
                print("❌ Could not connect to Gmail")
                return
        else:
            print("\n⚠️  Running without Gmail connection")
        
        # Main loop
        while True:
            print("\n" + "="*60)
            print("📋 Menu")
            print("="*60)
            print("  1. Browse and select emails to save")
            print("  2. Fetch recent emails (last 30 days)")
            print("  3. Search emails and save")
            print("  4. View current memories")
            print("  5. Ask about memories")
            print("  6. Quit")
            
            choice = input("\nEnter choice (1-6): ").strip()
            
            if choice == "1":
                # Browse and select
                selected_emails = self.browse_emails_interactive()
                if selected_emails:
                    result = self.create_memories_from_emails(selected_emails)
                    print(f"\n✅ {result.get('message', 'Done')}")
            
            elif choice == "2":
                # Fetch recent emails
                emails = self.fetch_recent_emails(max_results=50, days_back=30)
                if emails:
                    # Ask if user wants to save all or select
                    print(f"\n📧 Found {len(emails)} recent emails")
                    save_choice = input("Save all as memories? (y/n): ").strip().lower()
                    if save_choice == 'y':
                        result = self.create_memories_from_emails(emails)
                        print(f"\n✅ {result.get('message', 'Done')}")
            
            elif choice == "3":
                # Search and save
                selected_emails = self.browse_emails_interactive()
                if selected_emails:
                    result = self.create_memories_from_emails(selected_emails)
                    print(f"\n✅ {result.get('message', 'Done')}")
            
            elif choice == "4":
                # View memories
                souvenirs = self.souvenir_assistant.list_souvenirs(limit=20)
                print(f"\n📋 Stored Memories ({len(souvenirs)}):\n")
                for i, s in enumerate(souvenirs, 1):
                    print(f"{i}. {s['title']}")
                    print(f"   {s['content'][:100]}...")
                    print(f"   🏷️  {', '.join(s.get('tags', []))}")
                    print()
            
            elif choice == "5":
                # Ask about memories
                question = input("\n❓ Ask a question about your memories: ").strip()
                if question:
                    result = self.souvenir_assistant.ask_about_souvenirs(question)
                    if result.get('ok'):
                        print(f"\n💬 Answer:\n{result.get('answer', 'No answer')}")
                        if result.get('sources'):
                            print(f"\n📚 Sources: {', '.join(result.get('sources', []))}")
            
            elif choice == "6":
                print("\n👋 Goodbye!")
                break
            
            else:
                print("Invalid choice")
        
        # Cleanup
        self.souvenir_assistant.close()


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Gmail Memory Assistant - Create memories from your emails"
    )
    parser.add_argument(
        "--db",
        default="souvenir_memory.sqlite",
        help="Path to souvenir database"
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick mode: fetch recent emails and create memories"
    )
    
    args = parser.parse_args()
    
    assistant = GmailMemoryAssistant(souvenir_db=args.db)
    
    if args.quick:
        # Quick mode: connect and fetch
        assistant.credentials = assistant.get_credentials_interactive()
        if assistant.credentials:
            assistant.connect()
            emails = assistant.fetch_recent_emails(max_results=30)
            if emails:
                result = assistant.create_memories_from_emails(emails)
                print(f"\n✅ {result.get('message', 'Done')}")
        assistant.souvenir_assistant.close()
    else:
        # Interactive mode
        assistant.run_interactive()


if __name__ == "__main__":
    main()
