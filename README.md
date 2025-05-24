# Parcify - Automated Web Data Extractor

Parcify is a powerful web scraping tool that allows you to extract various types of data from websites with ease. It provides a user-friendly interface and supports multiple data extraction types.

## Features

- Extract different types of data:
  - Headings (H1)
  - Links with text
  - Images with alt text
  - Prices
  - Tables
- Modern, responsive user interface
- Export data in JSON and CSV formats
- Detailed error handling and logging
- Processing time tracking

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/parcify.git
cd parcify
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Start the application:
```bash
python app.py
```

2. Open your web browser and navigate to:
```
http://localhost:5000
```

3. Enter a URL and select the type of data you want to extract.

4. Click "Extract Data" and wait for the results.

5. Export the results in your preferred format (JSON or CSV).

## Development

### Project Structure
```
parcify/
├── app.py              # Main application file
├── requirements.txt    # Python dependencies
├── templates/          # HTML templates
│   └── index.html     # Main interface
└── parcify.log        # Application logs
```

### Adding New Features

1. Fork the repository
2. Create a new branch
3. Make your changes
4. Submit a pull request

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Support

If you encounter any issues or have questions, please open an issue in the GitHub repository. 