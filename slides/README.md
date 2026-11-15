# Presentation deck

- `deck.pdf`: the exported slides.
- `deck.html`: the same deck as a [reveal.js](https://revealjs.com) presentation. Open it in Chrome. Use ← → to move, F for fullscreen, Esc for the overview. The cover plays the demo video.

Re-export the PDF after editing `deck.html`:

```sh
"/Applications/Google Chrome.app/Contents/MacOS/Google Chrome" --headless=new --no-pdf-header-footer \
  --print-to-pdf=deck.pdf "file://$PWD/deck.html?pdf"
```

reveal.js (MIT) is vendored in `vendor/reveal.js/` with its licence.
