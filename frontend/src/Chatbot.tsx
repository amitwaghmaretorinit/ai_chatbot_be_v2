import React, { useState } from 'react';
import axios from 'axios';

const Chatbot: React.FC = () => {
  const [userMessage, setUserMessage] = useState<string>('');
  const [chatHistory, setChatHistory] = useState<{ sender: string; message: string }[]>([]);

  const handleMessageSend = async () => {
    if (userMessage.trim()) {
      setChatHistory([...chatHistory, { sender: 'user', message: userMessage }]);

      try {
        const response = await axios.post('http://127.0.0.1:5000/message', { message: userMessage });
        setChatHistory((prev) => [...prev, { sender: 'bot', message: response.data.response }]);
      } catch (error) {
        console.error('Error sending message:', error);
      } finally {
        setUserMessage('');
      }
    }
  };

  return (
    <div>
      <h1>Chatbot</h1>
      <div style={{ border: '1px solid #ccc', padding: '10px', height: '400px', overflowY: 'scroll' }}>
        {chatHistory.map((chat, index) => (
          <div key={index} style={{ textAlign: chat.sender === 'user' ? 'right' : 'left' }}>
            <strong>{chat.sender === 'user' ? 'You' : 'Bot'}:</strong> {chat.message}
          </div>
        ))}
      </div>
      <input
        type="text"
        value={userMessage}
        onChange={(e) => setUserMessage(e.target.value)}
        placeholder="Type your message..."
      />
      <button onClick={handleMessageSend}>Send</button>
    </div>
  );
};

export default Chatbot;
