import os
import requests
from dotenv import load_dotenv
from rich.console import Console
import torch

load_dotenv()
console = Console()



def log_error(message, webhook=True):
    console.log(message, style='red')
    if webhook:
        log_to_webhook(message)

def log_info(message, webhook=True):
    console.log(message, style='blue')
    if webhook:
        log_to_webhook(message)

def log_warning(message, webhook=True):
    console.log(message, style='yellow')
    if webhook:
        log_to_webhook(message)

def log_to_webhook(message):
    webhook = os.environ.get('LOG_WEBHOOK')
    if webhook is None:
        log_error("LOG_WEBHOOK env variable not set, but webhook log requested. This is a fatal error.", webhook=False)
    requests.post(webhook, {
        'content': message
    })

torch_device = 'cuda' if torch.cuda.is_available() else 'cpu'
log_info(f"Using torch device '{torch_device}'")
torch_device = torch.device(torch_device)