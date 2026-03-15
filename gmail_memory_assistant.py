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
import base64
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
MAILS_FILE = 'mails.pickle'


class GmailMemoryAssistant:
    """Gmail-powered memory assistant using souvenir storage."""
    
    def __init__(self, souvenir_db: str = "souvenir_memory.sqlite"):
        """Initialize Gmail memory assistant."""
        self.service = None
        self.credentials = None
        self.souvenir_assistant = SouvenirAssistant(db_path=souvenir_db, enable_embeddings=True)
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

        prev_tocken = self._load_existing_token()
        if prev_tocken is not None:
            return prev_tocken
        
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
    
    def fetch_recent_emails(self, max_results: int = 200, days_back: int = 90) -> List[Dict]:
        """Fetch recent emails from Gmail with increased history.
        
        Args:
            max_results: Maximum number of emails to fetch (default: 200)
            days_back: How many days back to fetch (default: 90 = 3 months)
        
        Returns:
            List of email dictionaries with extracted information
        """
        return self._fetch_emails_from_multiple_sources(max_results, days_back)
    
    def _fetch_emails_from_multiple_sources(self, max_results: int = 200, days_back: int = 90, skip_existing: bool = True) -> List[Dict]:
        """Fetch emails from multiple Gmail labels/folders.
        
        Fetches from: INBOX, SENT, and important categories to get comprehensive history.
        
        Args:
            max_results: Maximum emails per folder
            days_back: Days back to search
            skip_existing: If True, skip threads that already exist in database
            
        Returns:
            Combined list of all emails
        """
        if not self.service:
            print("❌ Not connected to Gmail")
            return []
        
        # Load stored thread_ids for deduplication if skip_existing is enabled
        stored_thread_ids = set()
        if skip_existing:
            print("   Loading stored thread IDs for deduplication...")
            stored_thread_ids = self._get_stored_thread_ids()
            print(f"   Found {len(stored_thread_ids)} existing threads in database")
        
        # Define which labels/folders to fetch from
        sources = [
            ('INBOX', 'inbox'),
            ('SENT', 'sent'),
            ('IMPORTANT', 'important'),
            ('STARRED', 'starred'),
        ]
        
        all_emails = []
        seen_ids = set()  # Avoid duplicates
        
        try:
            # Calculate date filter
            date_since = (datetime.now() - timedelta(days=days_back)).strftime('%Y/%m/%d')
            
            print(f"\n📬 Fetching emails from last {days_back} days ({max_results} per source)...")
            
            for label_id, label_name in sources:
                try:
                    print(f"\n  📂 Fetching from {label_name}...")
                    
                    # Get messages for this label
                    results = self.service.users().messages().list(
                        userId='me',
                        maxResults=max_results,
                        q=f'after:{date_since}',
                        labelIds=[label_id] if label_id != 'INBOX' else None
                    ).execute()
                    
                    messages = results.get('messages', [])
                    print(f"     Found {len(messages)} emails in {label_name}")
                    
                    # Fetch details for each message
                    skipped_existing = 0
                    for i, msg in enumerate(messages):
                        if msg['id'] in seen_ids:
                            continue
                        
                        # Check if thread already exists in database (to save API call)
                        if skip_existing and stored_thread_ids:
                            thread_id = msg.get('threadId')
                            if thread_id and thread_id in stored_thread_ids:
                                skipped_existing += 1
                                continue
                        
                        try:
                            message = self.service.users().messages().get(
                                userId='me',
                                id=msg['id'],
                                format='full'
                            ).execute()
                            
                            # Extract key info
                            email_data = self._extract_email_info(message)
                            all_emails.append(email_data)
                            seen_ids.add(msg['id'])
                            
                            if (i + 1) % 25 == 0:
                                print(f"     Processed {i + 1}/{len(messages)}...")
                                
                        except HttpError as e:
                            print(f"⚠️  Error fetching email {msg['id']}: {e}")
                            continue
                    
                    if skipped_existing > 0:
                        print(f"     Skipped {skipped_existing} existing threads (already in database)")
                    
                except HttpError as e:
                    print(f"⚠️  Error fetching {label_name}: {e}")
                    continue
            
            # Sort all emails by date (newest first)
            all_emails.sort(key=lambda e: e.get('date', ''), reverse=True)
            
            # Deduplicate by message ID, keeping first (newest)
            seen_ids = set()
            unique_emails = []
            for email in all_emails:
                msg_id = email.get('message_id', email.get('id', ''))
                if msg_id not in seen_ids:
                    unique_emails.append(email)
                    seen_ids.add(msg_id)
            
            print(f"\n✅ Successfully processed {len(unique_emails)} unique emails")
            return unique_emails
            
        except HttpError as e:
            print(f"❌ Error fetching emails: {e}")
            return []
    
    def _get_stored_thread_ids(self) -> Set[str]:
        """Get all thread_ids already stored in the database.
        
        Returns:
            Set of thread_id strings from the database
        """
        try:
            return self.souvenir_assistant.get_stored_thread_ids()
        except Exception as e:
            print(f"Warning: Failed to get stored thread IDs: {e}")
            return set()
    
    def fetch_comprehensive_history(self, max_results_per_source: int = 200, days_back: int = 365, skip_existing: bool = True) -> List[Dict]:
        """Fetch comprehensive email history across all major folders.
        
        This method fetches the maximum amount of email history for creating
        rich memories. It pulls from:
        - Inbox (personal communications)
        - Sent (outgoing emails)  
        - Important (marked important)
        - Starred (starred messages)
        - All Categories
        
        This method uses iterative fetching to bypass Gmail's 500 message limit
        per request by using page tokens for pagination.
        
        Args:
            max_results_per_source: Maximum emails to fetch (default: 200)
            days_back: How far back to search (default: 365 = 1 year)
            skip_existing: If True, skip threads that already exist in database
            
        Returns:
            List of unique email dictionaries
        """
        if not self.service:
            print("❌ Not connected to Gmail")
            return []
        
        # Load stored thread_ids for deduplication if skip_existing is enabled
        stored_thread_ids = set()
        if skip_existing:
            print("   Loading stored thread IDs for deduplication...")
            stored_thread_ids = self._get_stored_thread_ids()
            print(f"   Found {len(stored_thread_ids)} existing threads in database")
        
        all_emails = []
        seen_ids = set()
        
        try:
            date_since = (datetime.now() - timedelta(days=days_back)).strftime('%Y/%m/%d')
            
            print(f"\n📬 Fetching comprehensive email history...")
            print(f"   Max results per source: {max_results_per_source}")
            print(f"   Days back: {days_back}")
            
            try:
                print(f"\n  📂 Fetching...")
                
                query = f'after:{date_since}'
                
                # Gmail API has a hard limit of 500 messages per request
                # Use the smaller of max_results_per_source or 500
                batch_size = min(max_results_per_source, 500)
                
                # Initial request
                request_params = {
                    'userId': 'me',
                    'maxResults': batch_size,
                    'q': query,
                }
                results = self.service.users().messages().list(**request_params).execute()
                
                messages = results.get('messages', [])
                page_token = results.get('nextPageToken')
                
                print(f"     Retrieved first batch: {len(messages)} emails")
                
                # Iteratively fetch more pages until we reach max_results_per_source
                iteration = 1
                while page_token and len(seen_ids) < max_results_per_source:
                    remaining = max_results_per_source - len(seen_ids)
                    if remaining <= 0:
                        break
                    
                    # Request next page
                    request_params['pageToken'] = page_token
                    request_params['maxResults'] = min(remaining, 500)
                    
                    try:
                        results = self.service.users().messages().list(**request_params).execute()
                        batch_messages = results.get('messages', [])
                        messages.extend(batch_messages)
                        page_token = results.get('nextPageToken')
                        iteration += 1
                        print(f"     Retrieved batch {iteration}: {len(batch_messages)} emails (total: {len(messages)})")
                    except HttpError as e:
                        print(f"⚠️  Error fetching page {iteration}: {e}")
                        break
                
                print(f"     Found {len(messages)} emails in {iteration} batch(es)")
                
                # Fetch details (with batching for efficiency)
                detail_batch_size = 25
                skipped_existing = 0
                for i in range(0, len(messages), detail_batch_size):
                    batch = messages[i:i + detail_batch_size]
                    
                    for msg in batch:
                        if msg['id'] in seen_ids:
                            continue
                        
                        # Check if thread already exists in database (to save API call)
                        if skip_existing and stored_thread_ids:
                            thread_id = msg.get('threadId')
                            if thread_id and thread_id in stored_thread_ids:
                                skipped_existing += 1
                                continue
                        
                        try:
                            message = self.service.users().messages().get(
                                userId='me',
                                id=msg['id'],
                                format='full'
                            ).execute()
                            
                            email_data = self._extract_email_info(message)
                            msg_body = email_data.get('body')
                            assert len(msg_body) > 0
                            all_emails.append(email_data)
                            seen_ids.add(msg['id'])
                            
                        except HttpError:
                            continue
                    
                    if (i + detail_batch_size) % 50 == 0:
                        print(f"     Processed {min(i + detail_batch_size, len(messages))}/{len(messages)}...")
                
                if skipped_existing > 0:
                    print(f"     Skipped {skipped_existing} existing threads (already in database)")
                
            except HttpError as e:
                print(f"⚠️  Error fetching mails: {e}")
                pass
            
            # Sort by date
            all_emails.sort(key=lambda e: e.get('date', ''), reverse=True)
            
            # Final deduplication
            seen_msg_ids = set()
            unique_emails = []
            for email in all_emails:
                msg_id = email.get('message_id', email.get('id', ''))
                if msg_id and msg_id not in seen_msg_ids:
                    unique_emails.append(email)
                    seen_msg_ids.add(msg_id)
            
            print(f"\n✅ Total: {len(unique_emails)} unique emails fetched")
            return unique_emails
            
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
        
        # Extract thread-related headers for conversation reconstruction
        thread_id = message.get('threadId', '')
        message_id = header_dict.get('message-id', '')
        references = header_dict.get('references', '')
        in_reply_to = header_dict.get('in-reply-to', '')
        
        # Extract additional meaningful headers
        cc = header_dict.get('cc', '')
        bcc = header_dict.get('bcc', '')
        reply_to = header_dict.get('reply-to', '')
        
        return {
            'id': message['id'],
            'thread_id': thread_id,
            'subject': subject,
            'sender': sender,
            'to': to,
            'cc': cc,
            'bcc': bcc,
            'reply_to': reply_to,
            'date': date,
            'body': body,
            'labels': labels,
            'snippet': message.get('snippet', ''),
            'message_id': message_id,
            'references': references,
            'in_reply_to': in_reply_to
        }
    def _decode_data(self, data):
        return base64.urlsafe_b64decode(data).decode('utf-8')

    def _extract_body(self, payload: Dict) -> str:
        def walk(parts):
            for part in parts:
                mime_type = part.get('mimeType')
                
                # priority to plain text:
                if mime_type == 'text/plain':
                    body = self._decode_data(part['body']['data'])
                    if len(body) > 0:
                        return body
                
                if mime_type == 'text/html':
                    body = self._decode_data(part['body']['data'])
                    if len(body) > 0:
                        return body

                if 'parts' in part:
                    body = walk(part['parts'])
                    if len(body) > 0:
                        return body
            return ""
            
        return walk([payload])
    
    def _build_email_thread_tree(self, emails: List[Dict]) -> Dict[str, Any]:
        """
        Build a tree structure of email threads from a list of emails.
        Uses thread ID and References/In-Reply-To headers to reconstruct conversations.
        """
        from collections import defaultdict
        
        # Group emails by thread_id
        thread_map = defaultdict(list)
        for email in emails:
            thread_id = email.get('thread_id', '')
            if thread_id:
                thread_map[thread_id].append(email)
        
        # Sort each thread by date
        for thread_id in thread_map:
            thread_map[thread_id].sort(key=lambda e: e.get('date', ''))
        
        # Build conversation trees for each thread
        threads = {}
        for thread_id, thread_emails in thread_map.items():
            if len(thread_emails) > 1:
                # This is a conversation thread - build tree structure
                root_emails = []
                replies = defaultdict(list)
                
                # Parse references to build parent-child relationships
                for email in thread_emails:
                    refs = email.get('references', '').split()
                    in_reply_to = email.get('in_reply_to', '').strip('<>')
                    
                    # Find parent
                    parent_found = False
                    for ref in refs:
                        ref = ref.strip('<>')
                        if ref:
                            for potential_parent in thread_emails:
                                if potential_parent.get('message_id', '').strip('<>') == ref:
                                    replies[potential_parent['id']].append(email)
                                    parent_found = True
                                    break
                    
                    if not parent_found:
                        root_emails.append(email)
                
                # If no roots found by references, use first email as root
                if not root_emails and thread_emails:
                    root_emails = [thread_emails[0]]
                    for email in thread_emails[1:]:
                        replies[thread_emails[0]['id']].append(email)
                
                threads[thread_id] = {
                    'emails': thread_emails,
                    'root_emails': root_emails,
                    'replies': dict(replies),
                    'size': len(thread_emails)
                }
        
        return threads
    
    def _format_thread_for_memory(self, thread: Dict, emails: List[Dict]) -> str:
        """Format a thread of emails into a readable memory summary."""
        if not thread or not emails:
            return ""
        
        lines = []
        root_emails = thread.get('root_emails', [])
        replies = thread.get('replies', {})
        
        # Get conversation subject from first email
        subject = emails[0].get('subject', '(No Subject)')
        lines.append(f"📧 EMAIL CONVERSATION: {subject}")
        lines.append(f"Thread size: {len(emails)} messages")
        lines.append("")
        
        def format_email_tree(email, depth=0):
            indent = "  " * depth
            prefix = "└─ " if depth > 0 else ""
            
            sender = email.get('sender', 'Unknown')
            date = email.get('date', '')
            body = email.get('body', '')[:500]  # Limit body length
            
            # Extract first meaningful part of body
            body_preview = body.split('\n')[0] if body else ''
            
            lines.append(f"{indent}{prefix}From: {sender}")
            lines.append(f"{indent}{prefix}Date: {date}")
            if body_preview:
                lines.append(f"{indent}{prefix}Content: {body_preview}")
            lines.append("")
            
            # Process replies
            for reply in replies.get(email.get('id'), []):
                format_email_tree(reply, depth + 1)
        
        # Format each root email and its replies
        for root in root_emails:
            format_email_tree(root)
        
        return "\n".join(lines)
    
    def create_memories_from_emails(self, emails: List[Dict], include_single_emails: bool = True, store_individual_emails: bool = True) -> Dict[str, Any]:
        """Create souvenirs from extracted email data.
        
        Args:
            emails: List of email dictionaries
            include_single_emails: If True, also store single emails (not just threads)
            store_individual_emails: If True, store each email within threads separately (in addition to thread-level memory)
        
        Processes email threads (conversations with multiple messages) and optionally
        single emails (messages with no replies). Uses conversation_info from LLM 
        extraction to build rich memory content.
        """
        if not emails:
            return {"ok": False, "error": "No emails to process"}
        
        print(f"\n📝 Creating memories from {len(emails)} emails...")
        
        # Use SouvenirAssistant's thread builder to group related emails
        print("🔍 Building email conversation threads...")
        threads = self.souvenir_assistant.build_threads(
            emails, 
            use_subject_matching=False,
            use_body_similarity=False
        )
        
        # Only count threads with more than 1 email
        thread_count = sum(1 for t in threads.values() if t['size'] > 1)
        print(f"   Found {thread_count} email conversations (threads with multiple messages)")
        
        # Identify single emails (emails not part of any thread with multiple messages)
        single_emails = []
        for email in emails:
            thread_id = email.get('thread_id')
            if thread_id and thread_id in threads:
                if threads[thread_id]['size'] == 1:
                    # This is a single email in its thread
                    single_emails.append(email)
            else:
                # No thread ID at all - treat as single
                single_emails.append(email)
        
        if include_single_emails:
            print(f"   Found {len(single_emails)} single emails (not part of threads)")
        else:
            print(f"   Skipping {len(single_emails)} single emails (include_single_emails=False)")
        
        # Step 8: Email-level granularity option
        if store_individual_emails and thread_count > 0:
            print(f"   Will store individual emails within threads (store_individual_emails=True)")
        
        if thread_count == 0 and len(single_emails) == 0:
            return {
                "ok": True,
                "memories_created": 0,
                "emails_processed": len(emails),
                "threads_processed": 0,
                "single_emails_processed": 0,
                "message": f"No conversation threads or single emails found."
            }
        
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
        
        # Process threads (conversations with multiple emails)
        for thread_id, thread in threads.items():
            if thread['size'] <= 1:
                continue  # Skip single emails here, handle below
            
            thread_emails = thread['emails']
            
            # Use LLM to extract meaningful information from the conversation
            conversation_info = self.souvenir_assistant.process_conversation_for_memory(thread_emails)
            
            # Determine category based on labels from all emails in thread
            category = ''
            for email in thread_emails:
                if category:
                    break
                for label in email.get('labels', []):
                    if label in categories_map:
                        category = categories_map[label]
                        break
            if not category:
                category = 'email_conversation'
            
            # Use tags from LLM extraction, add labels
            tags = conversation_info.get("tags", ["conversation"])
            for email in thread_emails:
                for label in email.get('labels', []):
                    low_label = label.lower()
                    if low_label not in tags:
                        tags.append(low_label)
            tags = list(set(tags))[:15]
            
            # Add thread memory using new multi-chunk method
            result = self.souvenir_assistant.add_email_memory(
                thread_emails=thread_emails,
                conversation_info=conversation_info,
                category=category,
                tags=tags,
                thread_id=thread_id
            )
            
            if result.get('ok'):
                memories_created += 1
            
            # Step 8: Store individual emails within thread if enabled
            if store_individual_emails:
                for i, email in enumerate(thread_emails):
                    # Create individual email memory with unique content
                    individual_info = {
                        "topic": email.get('subject', 'No Subject'),
                        "participants": [email.get('from', 'Unknown'), email.get('to', '')],
                        "key_points": [email.get('snippet', '')] if email.get('snippet') else [],
                        "action_items": [],
                        "dates": [],
                        "entities": [],
                        "follow_ups": [],
                        "tags": ["individual_email", "thread_email"]
                    }
                    
                    # Determine category for individual email
                    ind_category = category
                    for label in email.get('labels', []):
                        if label in categories_map:
                            ind_category = categories_map[label]
                            break
                    
                    # Add tags for individual email
                    ind_tags = ["individual_email", "thread_email"]
                    for label in email.get('labels', []):
                        low_label = label.lower()
                        if low_label not in ind_tags:
                            ind_tags.append(low_label)
                    ind_tags = list(set(ind_tags))[:15]
                    
                    # Create individual memory - use lower threshold to allow storing
                    ind_result = self.souvenir_assistant.add_email_memory(
                        thread_emails=[email],
                        conversation_info=individual_info,
                        category=ind_category,
                        tags=ind_tags,
                        is_single_email=False,  # Use thread threshold (0.75) not single email (0.70)
                        thread_id=thread_id
                    )
                    
                    if ind_result.get('ok'):
                        memories_created += 1
        
        # Step 6: Process single emails if enabled
        single_emails_processed = 0
        individual_emails_stored = 0
        
        if include_single_emails and single_emails:
            print(f"   Processing {len(single_emails)} single emails...")
            for email in single_emails:
                # Get thread_id from email if available (may be None for true single emails)
                email_thread_id = email.get('thread_id')
                # Create a simple memory for single email
                single_result = self._create_single_email_memory(email, categories_map, email_thread_id)
                if single_result.get('ok'):
                    memories_created += 1
                    single_emails_processed += 1
        
        # Count individual emails stored (done inside the thread processing)
        # Already counted in memories_created
        
        return {
            "ok": True,
            "memories_created": memories_created,
            "emails_processed": len(emails),
            "threads_processed": thread_count,
            "single_emails_processed": single_emails_processed,
            "individual_emails_stored": store_individual_emails and thread_count > 0,
            "message": f"Created {memories_created} memories from {len(emails)} emails ({thread_count} threads, {single_emails_processed} single emails, individual emails stored: {store_individual_emails})"
        }
    
    def _create_single_email_memory(self, email: Dict, categories_map: Dict, thread_id: Optional[str] = None) -> Dict[str, Any]:
        """Create a memory from a single email (not part of a thread).
        
        Args:
            email: Single email dictionary
            categories_map: Mapping of Gmail labels to categories
            thread_id: Optional Gmail thread ID for deduplication tracking
            
        Returns:
            Result dictionary from add_email_memory
        """
        # Extract basic info from the email
        subject = email.get('subject', 'No Subject')
        sender = email.get('from', 'Unknown')
        snippet = email.get('snippet', '')
        
        # Create a simple conversation_info structure for single email
        conversation_info = {
            "topic": subject,
            "participants": [sender],
            "key_points": [snippet] if snippet else [],
            "action_items": [],
            "dates": [],
            "entities": [],
            "follow_ups": [],
            "tags": ["single_email", "email"]
        }
        
        # Determine category from labels
        category = 'email_single'
        for label in email.get('labels', []):
            if label in categories_map:
                category = categories_map[label]
                break
        
        # Add label tags
        tags = ["single_email", "email"]
        for label in email.get('labels', []):
            low_label = label.lower()
            if low_label not in tags:
                tags.append(low_label)
        tags = list(set(tags))[:15]
        
        # Create single email memory using add_email_memory with single email
        return self.souvenir_assistant.add_email_memory(
            thread_emails=[email],
            conversation_info=conversation_info,
            category=category,
            tags=tags,
            is_single_email=True,
            thread_id=thread_id
        )
    
    def analyze_email_memories(self) -> Dict[str, Any]:
        """Analyze all email memories to extract high-level insights.
        
        Now uses LLM-based analysis from SouvenirAssistant instead of hard-coded patterns.
        
        Returns:
            Dictionary containing analysis results
        """
        print("\n🔍 Analyzing email memories with LLM...")
        
        # Use LLM-based insights extraction
        result = self.souvenir_assistant.extract_insights_from_memories(
            category_filter='email',
            limit=100
        )
        
        if not result.get("ok"):
            return {
                "ok": False,
                "error": result.get("error", "Failed to analyze memories")
            }
        
        insights = result.get("insights", {})
        
        return {
            "ok": True,
            "analysis": {
                "total_memories_analyzed": result.get("memories_analyzed", 0),
                "key_themes": insights.get("key_themes", []),
                "important_contacts": insights.get("important_contacts", []),
                "pending_actions": insights.get("pending_actions", []),
                "upcoming_events": insights.get("upcoming_events", []),
                "summary": insights.get("summary", "")
            }
        }
    
    def answer_email_question(self, question: str) -> Dict[str, Any]:
        """Answer high-level questions about email memories using LLM.
        
        Uses LLM-based analysis from SouvenirAssistant.
        
        Examples:
        - "Which person sends me the most emails?"
        - "Which companies send me emails?"
        - "What pending appointments do I have?"
        - "What action items are there?"
        
        Args:
            question: Natural language question about emails
            
        Returns:
            Answer dictionary with results
        """
        print(f"\n🔍 Answering: {question}")
        
        # Use LLM-based analysis from SouvenirAssistant
        result = self.souvenir_assistant.analyze_memories(
            question=question,
            category_filter='email',
            limit=50
        )
        
        if not result.get("ok"):
            return {
                "ok": False,
                "error": result.get("error", "Failed to answer question")
            }
        
        return {
            "ok": True,
            "answer": result.get("answer", ""),
            "sources": result.get("sources", [])
        }
    
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
            print("  2. Fetch recent emails (last 90 days, 200 emails)")
            print("  3. Fetch comprehensive history (1 year, all folders)")
            print("  4. Search emails and save")
            print("  5. View current memories")
            print("  6. Ask about memories")
            print("  7. Get summary of memory")
            print("  8. Quit")
            
            choice = input("\nEnter choice (1-8): ").strip()
            
            if choice == "1":
                # Browse and select
                selected_emails = self.browse_emails_interactive()
                if selected_emails:
                    result = self.create_memories_from_emails(selected_emails)
                    print(f"\n✅ {result.get('message', 'Done')}")
            
            elif choice == "2":
                # Fetch recent emails (new improved method)
                emails = self.fetch_recent_emails(max_results=100, days_back=90)
                if emails:
                    # Ask if user wants to save all or select
                    print(f"\n📧 Found {len(emails)} recent emails")
                    save_choice = input("Save all as memories? (y/n): ").strip().lower()
                    if save_choice == 'y':
                        result = self.create_memories_from_emails(emails)
                        print(f"\n✅ {result.get('message', 'Done')}")
            
            elif choice == "3":
                # Fetch comprehensive history (new option)
                print("\n📬 This will fetch emails from all your Gmail folders")
                print("   (Inbox, Sent, Important, Starred, Categories)")
                print("   Going back up to 1 year...")
                confirm = input("Continue? (y/n): ").strip().lower()
                if confirm == 'y':
                    if os.path.exists(MAILS_FILE):
                        with open(MAILS_FILE, 'rb') as f:
                            emails = pickle.load(f)
                    else:
                        emails = self.fetch_comprehensive_history(
                            max_results_per_source=10000, 
                            days_back=365
                        )
                        
                        if emails:
                            with open(MAILS_FILE, 'wb') as f:
                                pickle.dump(emails, f)
                    if emails:
                        print(f"\n📧 Found {len(emails)} unique emails")
                        save_choice = input("Save all as memories? (y/n): ").strip().lower()
                        if save_choice == 'y':
                            result = self.create_memories_from_emails(emails)
                            print(f"\n✅ {result.get('message', 'Done')}")
            
            elif choice == "4":
                # Search and save
                selected_emails = self.browse_emails_interactive()
                if selected_emails:
                    result = self.create_memories_from_emails(selected_emails)
                    print(f"\n✅ {result.get('message', 'Done')}")
            
            elif choice == "5":
                # View memories
                souvenirs = self.souvenir_assistant.list_souvenirs(limit=20)
                print(f"\n📋 Stored Memories ({len(souvenirs)}):\n")
                for i, s in enumerate(souvenirs, 1):
                    print(f"{i}. {s['title']}")
                    print(f"   🏷️  {', '.join(s.get('tags', []))}")
                    print()
            
            elif choice == "6":
                # Ask about memories (general)
                question = input("\n❓ Ask a question about your memories: ").strip()
                if question:
                    result = self.souvenir_assistant.ask_about_souvenirs(question)
                    if result.get('ok'):
                        print(f"\n💬 Answer:\n{result.get('answer', 'No answer')}")
                        if result.get('sources'):
                            print(f"\n📚 Sources: {', '.join(result.get('sources', []))}")
            
            elif choice == "7":
                self.souvenir_assistant.ltm.print_cluster_summary(20, chunk_types=["email_original"])
            elif choice == "8":
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
        help="Quick mode: fetch recent emails (last 90 days, 200 emails) and create memories"
    )
    parser.add_argument(
        "--comprehensive",
        action="store_true",
        help="Comprehensive mode: fetch 1 year of emails from all folders for maximum memory creation"
    )
    parser.add_argument(
        "--max-results",
        type=int,
        default=200,
        help="Maximum number of emails to fetch per source (default: 200)"
    )
    parser.add_argument(
        "--days-back",
        type=int,
        default=90,
        help="How many days back to fetch emails (default: 90)"
    )
    
    args = parser.parse_args()
    
    assistant = GmailMemoryAssistant(souvenir_db=args.db)
    
    if args.quick or args.comprehensive:
        # Quick or comprehensive mode: connect and fetch
        assistant.credentials = assistant.get_credentials_interactive()
        if assistant.credentials:
            assistant.connect()
            
            if args.comprehensive:
                # Fetch comprehensive history (1 year, all folders)
                print("\n📬 Running comprehensive mode - fetching 1 year of email history...")
                emails = assistant.fetch_comprehensive_history(
                    max_results_per_source=args.max_results,
                    days_back=365
                )
            else:
                # Quick mode: fetch recent emails with better defaults
                emails = assistant.fetch_recent_emails(
                    max_results=args.max_results,
                    days_back=args.days_back
                )
            
            if emails:
                result = assistant.create_memories_from_emails(emails)
                print(f"\n✅ {result.get('message', 'Done')}")
            else:
                print("No emails found to process")
                
        assistant.souvenir_assistant.close()
    else:
        # Interactive mode
        assistant.run_interactive()


if __name__ == "__main__":
    main()
