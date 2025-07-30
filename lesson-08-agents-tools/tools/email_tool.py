#!/usr/bin/env python3
"""
Email Management Tool for Agents

Provides email composition, sending, and management capabilities
with safety controls and template support.
"""

import re
import json
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
from typing import Any, Optional, Dict, List, Union
from datetime import datetime
from pathlib import Path
from pydantic import BaseModel, Field
from langchain.tools import BaseTool


class EmailTool(BaseTool):
    """Comprehensive email management tool with safety controls."""
    
    name = "email"
    description = """
    Manage email operations including composition, sending, and templates.
    
    Operations:
    - compose: Create email with template
    - send: Send email (requires approval in production)
    - template: Create or use email template
    - validate: Validate email addresses
    - schedule: Schedule email for later (mock)
    
    Input format: "operation:parameters"
    Examples:
    - "compose:to=user@example.com,subject=Hello,body=Test message"
    - "template:name=welcome,type=create"
    - "validate:email=test@example.com"
    
    Returns: Operation result with email details
    """
    
    def __init__(self, 
                 smtp_server: str = "localhost",
                 smtp_port: int = 587,
                 smtp_username: Optional[str] = None,
                 smtp_password: Optional[str] = None,
                 safe_mode: bool = True):
        super().__init__()
        self.smtp_server = smtp_server
        self.smtp_port = smtp_port
        self.smtp_username = smtp_username
        self.smtp_password = smtp_password
        self.safe_mode = safe_mode  # In safe mode, emails are not actually sent
        
        # Email templates storage
        self.templates = self._load_default_templates()
        
        # Sent emails log (for tracking)
        self.sent_emails = []
        
        # Allowed domains for safety (in production)
        self.allowed_domains = ["example.com", "test.com", "localhost"]
        
        # Rate limiting
        self.rate_limit = 10  # emails per hour
        self.sent_count = 0
        self.rate_window_start = datetime.now()
    
    def _run(self, command: str) -> str:
        """Execute email operation."""
        try:
            if ":" not in command:
                return "Error: Invalid command format. Use 'operation:parameters'"
            
            operation, params_str = command.split(":", 1)
            operation = operation.strip().lower()
            
            # Parse parameters
            params = self._parse_parameters(params_str)
            
            # Route to appropriate operation
            if operation == "compose":
                return self._compose_email(params)
            elif operation == "send":
                return self._send_email(params)
            elif operation == "template":
                return self._manage_template(params)
            elif operation == "validate":
                return self._validate_email(params)
            elif operation == "schedule":
                return self._schedule_email(params)
            elif operation == "list":
                return self._list_sent_emails(params)
            else:
                return f"Error: Unknown operation '{operation}'. Available: compose, send, template, validate, schedule, list"
        
        except Exception as e:
            return f"Error executing email operation: {str(e)}"
    
    def _parse_parameters(self, params_str: str) -> Dict[str, str]:
        """Parse command parameters."""
        params = {}
        
        # Split by comma, but handle quoted values
        parts = []
        current_part = ""
        in_quotes = False
        
        for char in params_str:
            if char == '"' and (not current_part or current_part[-1] != '\\'):
                in_quotes = not in_quotes
            elif char == ',' and not in_quotes:
                if current_part:
                    parts.append(current_part)
                current_part = ""
                continue
            
            current_part += char
        
        if current_part:
            parts.append(current_part)
        
        # Parse key=value pairs
        for part in parts:
            if "=" in part:
                key, value = part.split("=", 1)
                params[key.strip()] = value.strip().strip('"')
        
        return params
    
    def _compose_email(self, params: Dict[str, str]) -> str:
        """Compose an email."""
        try:
            # Required parameters
            to_email = params.get("to", "")
            subject = params.get("subject", "")
            body = params.get("body", "")
            
            if not to_email:
                return "Error: 'to' parameter is required"
            
            # Validate email addresses
            validation_result = self._validate_email_address(to_email)
            if not validation_result["valid"]:
                return f"Error: Invalid email address: {to_email}"
            
            # Optional parameters
            from_email = params.get("from", "agent@example.com")
            cc = params.get("cc", "")
            bcc = params.get("bcc", "")
            template_name = params.get("template", "")
            priority = params.get("priority", "normal")
            
            # Use template if specified
            if template_name and template_name in self.templates:
                template = self.templates[template_name]
                if not subject:
                    subject = template.get("subject", "")
                if not body:
                    body = template.get("body", "")
                
                # Replace template variables
                body = self._replace_template_variables(body, params)
                subject = self._replace_template_variables(subject, params)
            
            # Create email composition
            email_data = {
                "id": f"email_{int(datetime.now().timestamp())}",
                "from": from_email,
                "to": to_email,
                "cc": cc,
                "bcc": bcc,
                "subject": subject,
                "body": body,
                "priority": priority,
                "created_at": datetime.now().isoformat(),
                "status": "composed"
            }
            
            # Format response
            result = f"✅ Email composed successfully\\n\\n"
            result += f"📧 **Email Details:**\\n"
            result += f"ID: {email_data['id']}\\n"
            result += f"From: {email_data['from']}\\n"
            result += f"To: {email_data['to']}\\n"
            
            if cc:
                result += f"CC: {cc}\\n"
            if bcc:
                result += f"BCC: {bcc}\\n"
            
            result += f"Subject: {subject}\\n"
            result += f"Priority: {priority}\\n\\n"
            result += f"📝 **Message Preview:**\\n"
            result += f"{body[:200]}{'...' if len(body) > 200 else ''}\\n\\n"
            result += f"📊 **Statistics:**\\n"
            result += f"Subject length: {len(subject)} characters\\n"
            result += f"Body length: {len(body)} characters\\n"
            
            if template_name:
                result += f"Template used: {template_name}\\n"
            
            result += f"\\n💡 Use 'send:id={email_data['id']}' to send this email"
            
            # Store composed email for sending
            self.sent_emails.append(email_data)
            
            return result
        
        except Exception as e:
            return f"Error composing email: {str(e)}"
    
    def _send_email(self, params: Dict[str, str]) -> str:
        """Send a composed email."""
        try:
            email_id = params.get("id", "")
            
            if not email_id:
                return "Error: Email ID is required for sending"
            
            # Find composed email
            email_data = None
            for email in self.sent_emails:
                if email.get("id") == email_id:
                    email_data = email
                    break
            
            if not email_data:
                return f"Error: Email with ID '{email_id}' not found"
            
            if email_data.get("status") == "sent":
                return f"Error: Email {email_id} has already been sent"
            
            # Check rate limiting
            if not self._check_rate_limit():
                return f"Error: Rate limit exceeded. Maximum {self.rate_limit} emails per hour"
            
            # Safety checks
            if self.safe_mode:
                # In safe mode, don't actually send
                result = self._simulate_email_send(email_data)
            else:
                # Actually send email (requires SMTP configuration)
                result = self._actually_send_email(email_data)
            
            # Update email status
            email_data["status"] = "sent"
            email_data["sent_at"] = datetime.now().isoformat()
            
            # Increment rate limit counter
            self.sent_count += 1
            
            return result
        
        except Exception as e:
            return f"Error sending email: {str(e)}"
    
    def _simulate_email_send(self, email_data: Dict) -> str:
        """Simulate email sending (safe mode)."""
        result = f"📧 Email sent successfully (SIMULATION MODE)\\n\\n"
        result += f"📋 **Delivery Report:**\\n"
        result += f"Email ID: {email_data['id']}\\n"
        result += f"From: {email_data['from']}\\n"
        result += f"To: {email_data['to']}\\n"
        result += f"Subject: {email_data['subject']}\\n"
        result += f"Sent at: {email_data['sent_at']}\\n"
        result += f"Status: ✅ Delivered (simulated)\\n\\n"
        result += f"⚠️ **Note:** This email was not actually sent due to safe mode being enabled."
        
        return result
    
    def _actually_send_email(self, email_data: Dict) -> str:
        """Actually send email via SMTP."""
        try:
            # Create message
            msg = MIMEMultipart()
            msg['From'] = email_data['from']
            msg['To'] = email_data['to']
            msg['Subject'] = email_data['subject']
            
            if email_data.get('cc'):
                msg['Cc'] = email_data['cc']
            
            # Add body
            msg.attach(MIMEText(email_data['body'], 'plain'))
            
            # Connect to SMTP server
            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                if self.smtp_username and self.smtp_password:
                    server.starttls()
                    server.login(self.smtp_username, self.smtp_password)
                
                # Send email
                text = msg.as_string()
                recipients = [email_data['to']]
                if email_data.get('cc'):
                    recipients.extend(email_data['cc'].split(','))
                if email_data.get('bcc'):
                    recipients.extend(email_data['bcc'].split(','))
                
                server.sendmail(email_data['from'], recipients, text)
            
            result = f"📧 Email sent successfully\\n\\n"
            result += f"📋 **Delivery Report:**\\n"
            result += f"Email ID: {email_data['id']}\\n"
            result += f"Recipients: {len(recipients)}\\n"
            result += f"Status: ✅ Delivered\\n"
            
            return result
        
        except Exception as e:
            return f"Error sending email via SMTP: {str(e)}"
    
    def _manage_template(self, params: Dict[str, str]) -> str:
        """Manage email templates."""
        try:
            action = params.get("action", "list")
            template_name = params.get("name", "")
            
            if action == "list":
                return self._list_templates()
            elif action == "create":
                return self._create_template(params)
            elif action == "get":
                return self._get_template(template_name)
            elif action == "delete":
                return self._delete_template(template_name)
            else:
                return f"Error: Unknown template action '{action}'. Available: list, create, get, delete"
        
        except Exception as e:
            return f"Error managing template: {str(e)}"
    
    def _list_templates(self) -> str:
        """List available email templates."""
        if not self.templates:
            return "📧 No email templates available"
        
        result = f"📧 Available Email Templates ({len(self.templates)})\\n"
        result += "=" * 50 + "\\n\\n"
        
        for name, template in self.templates.items():
            result += f"📄 **{name}**\\n"
            result += f"   Subject: {template.get('subject', 'No subject')}\\n"
            result += f"   Category: {template.get('category', 'General')}\\n"
            result += f"   Variables: {', '.join(template.get('variables', []))}\\n\\n"
        
        return result
    
    def _create_template(self, params: Dict[str, str]) -> str:
        """Create a new email template."""
        template_name = params.get("name", "")
        subject = params.get("subject", "")
        body = params.get("body", "")
        category = params.get("category", "General")
        
        if not template_name:
            return "Error: Template name is required"
        
        if not body:
            return "Error: Template body is required"
        
        # Extract variables from template
        variables = self._extract_template_variables(subject + " " + body)
        
        template = {
            "subject": subject,
            "body": body,
            "category": category,
            "variables": variables,
            "created_at": datetime.now().isoformat()
        }
        
        self.templates[template_name] = template
        
        result = f"✅ Email template '{template_name}' created successfully\\n\\n"
        result += f"📄 **Template Details:**\\n"
        result += f"Name: {template_name}\\n"
        result += f"Subject: {subject}\\n"
        result += f"Category: {category}\\n"
        result += f"Variables: {', '.join(variables)}\\n"
        result += f"Body length: {len(body)} characters\\n"
        
        return result
    
    def _get_template(self, template_name: str) -> str:
        """Get template details."""
        if not template_name:
            return "Error: Template name is required"
        
        if template_name not in self.templates:
            return f"Error: Template '{template_name}' not found"
        
        template = self.templates[template_name]
        
        result = f"📄 **Template: {template_name}**\\n"
        result += "=" * 40 + "\\n\\n"
        result += f"Subject: {template['subject']}\\n"
        result += f"Category: {template.get('category', 'General')}\\n"
        result += f"Variables: {', '.join(template.get('variables', []))}\\n"
        result += f"Created: {template.get('created_at', 'Unknown')}\\n\\n"
        result += f"**Body:**\\n{template['body']}"
        
        return result
    
    def _delete_template(self, template_name: str) -> str:
        """Delete a template."""
        if not template_name:
            return "Error: Template name is required"
        
        if template_name not in self.templates:
            return f"Error: Template '{template_name}' not found"
        
        del self.templates[template_name]
        return f"✅ Template '{template_name}' deleted successfully"
    
    def _validate_email(self, params: Dict[str, str]) -> str:
        """Validate email addresses."""
        email = params.get("email", "")
        
        if not email:
            return "Error: Email address is required"
        
        validation = self._validate_email_address(email)
        
        result = f"📧 Email Validation Report\\n"
        result += "=" * 40 + "\\n\\n"
        result += f"Email: {email}\\n"
        result += f"Valid: {'✅ Yes' if validation['valid'] else '❌ No'}\\n"
        
        if not validation['valid']:
            result += f"Issues: {', '.join(validation['issues'])}\\n"
        else:
            result += f"Domain: {validation['domain']}\\n"
            result += f"Local part: {validation['local']}\\n"
        
        return result
    
    def _schedule_email(self, params: Dict[str, str]) -> str:
        """Schedule email for later sending (mock implementation)."""
        email_id = params.get("id", "")
        send_time = params.get("time", "")
        
        if not email_id:
            return "Error: Email ID is required for scheduling"
        
        if not send_time:
            return "Error: Send time is required (format: YYYY-MM-DD HH:MM)"
        
        # Mock scheduling
        result = f"📅 Email scheduled successfully\\n\\n"
        result += f"📧 Email ID: {email_id}\\n"
        result += f"⏰ Scheduled for: {send_time}\\n"
        result += f"📊 Status: Scheduled\\n\\n"
        result += f"⚠️ **Note:** This is a mock implementation. In production, this would integrate with a job scheduler."
        
        return result
    
    def _list_sent_emails(self, params: Dict[str, str]) -> str:
        """List sent emails."""
        limit = int(params.get("limit", "10"))
        
        if not self.sent_emails:
            return "📧 No emails in history"
        
        result = f"📧 Email History (Last {min(limit, len(self.sent_emails))} emails)\\n"
        result += "=" * 60 + "\\n\\n"
        
        for email in self.sent_emails[-limit:]:
            status_icon = "✅" if email.get("status") == "sent" else "📝"
            result += f"{status_icon} **{email.get('id', 'Unknown')}**\\n"
            result += f"   To: {email.get('to', 'Unknown')}\\n"
            result += f"   Subject: {email.get('subject', 'No subject')}\\n"
            result += f"   Status: {email.get('status', 'Unknown')}\\n"
            result += f"   Created: {email.get('created_at', 'Unknown')}\\n\\n"
        
        return result
    
    def _validate_email_address(self, email: str) -> Dict[str, Any]:
        """Validate email address format and domain."""
        validation = {
            "valid": False,
            "issues": [],
            "domain": "",
            "local": ""
        }
        
        # Basic format validation
        email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}$'
        if not re.match(email_pattern, email):
            validation["issues"].append("Invalid email format")
            return validation
        
        # Split email
        try:
            local, domain = email.split('@')
            validation["local"] = local
            validation["domain"] = domain
        except ValueError:
            validation["issues"].append("Invalid email structure")
            return validation
        
        # Check local part
        if len(local) > 64:
            validation["issues"].append("Local part too long (max 64 characters)")
        
        if local.startswith('.') or local.endswith('.'):
            validation["issues"].append("Local part cannot start or end with a dot")
        
        # Check domain
        if len(domain) > 255:
            validation["issues"].append("Domain too long (max 255 characters)")
        
        # In safe mode, check allowed domains
        if self.safe_mode and domain not in self.allowed_domains:
            validation["issues"].append(f"Domain not in allowed list: {self.allowed_domains}")
        
        # If no issues, it's valid
        validation["valid"] = len(validation["issues"]) == 0
        
        return validation
    
    def _load_default_templates(self) -> Dict[str, Dict]:
        """Load default email templates."""
        return {
            "welcome": {
                "subject": "Welcome to {company_name}!",
                "body": """Hello {name},
                
Welcome to {company_name}! We're excited to have you on board.

Your account has been successfully created with the following details:
- Username: {username}
- Email: {email}

Next steps:
1. Complete your profile setup
2. Explore our features
3. Contact support if you need help

Best regards,
The {company_name} Team""",
                "category": "Onboarding",
                "variables": ["name", "company_name", "username", "email"]
            },
            "password_reset": {
                "subject": "Password Reset Request",
                "body": """Hello {name},

We received a request to reset your password. Click the link below to create a new password:

{reset_link}

This link will expire in 24 hours. If you didn't request this reset, please ignore this email.

Best regards,
Security Team""",
                "category": "Security",
                "variables": ["name", "reset_link"]
            },
            "notification": {
                "subject": "Important Notification: {subject}",
                "body": """Hello {name},

{message}

This is an automated notification from our system.

Best regards,
System Administrator""",
                "category": "System",
                "variables": ["name", "subject", "message"]
            }
        }
    
    def _extract_template_variables(self, text: str) -> List[str]:
        """Extract template variables from text."""
        pattern = r'\\{([^}]+)\\}'
        variables = re.findall(pattern, text)
        return list(set(variables))  # Remove duplicates
    
    def _replace_template_variables(self, text: str, params: Dict[str, str]) -> str:
        """Replace template variables with actual values."""
        for key, value in params.items():
            placeholder = f"{{{key}}}"
            text = text.replace(placeholder, value)
        
        return text
    
    def _check_rate_limit(self) -> bool:
        """Check if rate limit allows sending."""
        now = datetime.now()
        
        # Reset counter if hour has passed
        if (now - self.rate_window_start).total_seconds() > 3600:
            self.sent_count = 0
            self.rate_window_start = now
        
        return self.sent_count < self.rate_limit
    
    async def _arun(self, command: str) -> str:
        """Async version of email tool."""
        return self._run(command)


# Example usage and testing
def test_email_tool():
    """Test the email tool."""
    email_tool = EmailTool()
    
    test_commands = [
        "template:action=list",
        "validate:email=test@example.com",
        "validate:email=invalid-email",
        "compose:to=user@example.com,subject=Test Email,body=This is a test message",
        "template:action=create,name=test_template,subject=Test Subject,body=Hello {name}, this is a test.",
        "compose:to=user@example.com,template=welcome,name=John Doe,company_name=Test Corp,username=johndoe,email=john@example.com",
        "list:limit=5"
    ]
    
    print("📧 Email Tool Test Results:")
    print("=" * 60)
    
    for cmd in test_commands:
        result = email_tool._run(cmd)
        print(f"Command: {cmd}")
        print(f"Result:\\n{result}")
        print("-" * 40)


if __name__ == "__main__":
    test_email_tool()