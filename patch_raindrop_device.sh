#!/bin/bash
# Patch RainDrop models_rd.py to be device-agnostic
# Replaces .cuda() with .to(device)

set -e

FILE="src/models/raindrop/Raindrop/code/models_rd.py"

if [ ! -f "$FILE" ]; then
    echo "Error: $FILE not found"
    echo "Make sure you're in the project root directory"
    exit 1
fi

echo "Patching $FILE to be device-agnostic..."

# Backup original
cp "$FILE" "${FILE}.backup"
echo "✓ Created backup: ${FILE}.backup"

# Add device parameter to __init__ (line 62)
sed -i.tmp '62s/global_structure=None/global_structure=None, device=None/' "$FILE"

# Add self.device after super().__init__() (after line 63)
sed -i.tmp '63 a\
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
' "$FILE"

# Replace .cuda() with .to(self.device)
sed -i.tmp 's/\.cuda()/.to(self.device)/g' "$FILE"

# Clean up temp files
rm -f "${FILE}.tmp"

echo "✓ Patched successfully!"
echo ""
echo "Changes made:"
echo "  - Added 'device' parameter to Raindrop.__init__()"
echo "  - Added self.device attribute"
echo "  - Replaced all .cuda() with .to(self.device)"
echo ""
echo "To verify:"
echo "  grep -n '\.cuda()' $FILE"
echo "  (should return no results)"
echo ""
echo "To restore original:"
echo "  mv ${FILE}.backup $FILE"

