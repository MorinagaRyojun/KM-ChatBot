# SMEGPT

SMEGPT is a Streamlit-based chatbot application that provides information about SMEs (Small and Medium-sized Enterprises) and related events. It uses the Gemini AI model from Google to generate responses and incorporates context from web scraping to enhance its knowledge base.

## Features

- Interactive chat interface powered by Streamlit
- Integration with Google's Gemini AI model for natural language processing
- Web scraping to gather context about SMEs and events
- Automatic detection of event-related queries
- Context-aware responses that combine general SME information and event-specific details
- Chat history management
- Timeout handling for long-running queries

## Prerequisites

Before you begin, ensure you have met the following requirements:

- Python 3.7+
- A Gemini API key from Google

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/yourusername/smegpt.git
   cd smegpt
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Create a `.env` file in the project root and add your Gemini API key:
   ```
   GEMINI_API_KEY=your_api_key_here
   ```

## Usage

To run the application, use the following command:

```
streamlit run app.py
```

This will start the Streamlit server and open the application in your default web browser.

## Configuration

You can adjust the following parameters in the `app.py` file:

- `threshold` in `is_related_question()`: Determines how similar a new question must be to previous questions to be considered related.
- `chunk_size` and `chunk_overlap` in `get_general_answer_context()` and `get_event_information()`: Adjust these to change how text is split for processing.
- `threshold` in `is_event_query()`: Modifies the sensitivity of event-related query detection.

## Contributing

Contributions to SMEGPT are welcome. Please follow these steps to contribute:

1. Fork the repository
2. Create a new branch (`git checkout -b feature/your-feature-name`)
3. Make your changes
4. Commit your changes (`git commit -am 'Add some feature'`)
5. Push to the branch (`git push origin feature/your-feature-name`)
6. Create a new Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Streamlit for the web application framework
- Google for the Gemini AI model
- langchain for the LLM integration utilities

## Contact

If you have any questions or feedback, please open an issue in this repository.