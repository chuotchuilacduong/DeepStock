// Get necessary HTML elements
const chatBox = document.getElementById('chat-box');
const userInput = document.getElementById('user-input');
const sendBtn = document.getElementById('send-btn');
const loader = document.getElementById('loader');
const thinkingToggle = document.getElementById('thinking-toggle');
const suggestionsContainer = document.getElementById('prompt-suggestions'); // CẬP NHẬT: Lấy element mới

// !!! IMPORTANT: Change this URL to your API address
const API_URL = 'http://127.0.0.1:8000/chat';

// Function to send a message
const sendMessage = async () => {
    const userText = userInput.value.trim();
    if (userText === "") return;

    appendMessage(userText, 'user');
    userInput.value = '';
    loader.classList.remove('hidden');
    suggestionsContainer.innerHTML = ''; // CẬP NHẬT: Xóa các gợi ý sau khi gửi tin nhắn

    try {
        const response = await fetch(API_URL, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ message: userText }),
        });

        if (!response.ok) {
            const errorData = await response.json().catch(() => ({ detail: 'Could not read error response.' }));
            throw new Error(`Network error: ${response.status} - ${errorData.detail}`);
        }

        const data = await response.json();
        
        if (thinkingToggle.checked && data.thinking) {
            appendMessage(data.thinking, 'thinking');
        }
        
        appendMessage(data.answer, 'bot');

    } catch (error) {
        console.error('Error:', error);
        appendMessage(`Sorry, an error occurred: ${error.message}`, 'bot');
    } finally {
        loader.classList.add('hidden');
    }
};

/**
 * Appends a message to the chat box.
 * @param {string} text - The message content.
 * @param {'user' | 'bot' | 'thinking'} type - The type of message.
 */
function appendMessage(text, type) {
    const wrapper = document.createElement('div');
    wrapper.classList.add('message-wrapper', `${type}-message-wrapper`);

    let avatarContent = '';
    let messageClass = '';

    if (type === 'user') {
        avatarContent = '🧑‍💻';
        messageClass = 'user-message';
    } else if (type === 'bot') {
        avatarContent = '🤖';
        messageClass = 'bot-message';
    } else { // thinking
        avatarContent = '🤔';
        messageClass = 'thinking-message';
    }
    
    wrapper.innerHTML = `
        <div class="avatar">${avatarContent}</div>
        <div class="${messageClass}">
            <div class="message">${text}</div>
        </div>
    `;

    chatBox.appendChild(wrapper);
    chatBox.scrollTop = chatBox.scrollHeight;
}

/**
 * CẬP NHẬT: Hàm mới để tạo và hiển thị các nút gợi ý
 */
function renderSuggestions() {
    suggestionsContainer.innerHTML = ''; // Xóa các gợi ý cũ
    const suggestions = [
        "What is the latest news for Zoeis Inc.?",
        "Analyze the financial health of Zimmer Biomet Holdings, Inc.",
        "What does Zoetis Inc. (ZTS) primarily do?",
        "Does Zebra Technologies Corporation worth investing in?"
    ];

    suggestions.forEach(text => {
        const btn = document.createElement('button');
        btn.classList.add('suggestion-btn');
        btn.innerText = text;
        btn.addEventListener('click', () => {
            userInput.value = text; // Điền câu hỏi vào ô input
            sendMessage();           // Gửi tin nhắn
        });
        suggestionsContainer.appendChild(btn);
    });
}

// Event Listeners
sendBtn.addEventListener('click', sendMessage);
userInput.addEventListener('keypress', (event) => {
    if (event.key === 'Enter') {
        sendMessage();
    }
});

// Thêm tin nhắn chào mừng và hiển thị gợi ý khi tải trang
window.addEventListener('load', () => {
    appendMessage('Hello! How can I help you with the stock market today?', 'bot');
    renderSuggestions(); // CẬP NHẬT: Gọi hàm hiển thị gợi ý
});