#!/usr/bin/env python3
"""
Analyze character patterns in real sssalud CAPTCHAs to improve training
"""

def analyze_captcha_patterns():
    # Your real CAPTCHA examples
    real_captchas = [
        "UKhGh9",
        "26WanS", 
        "e4TkHP",
        "cGfFE2",
        "gnRYZe",
        "v76Ebu",
        "DUzp49",
        "MWR3mw",
        "3h2vUF",
        "t2md2m"
    ]
    
    uppercase_count = 0
    lowercase_count = 0
    digit_count = 0
    total_chars = 0
    
    char_positions = {i: {'upper': 0, 'lower': 0, 'digit': 0} for i in range(6)}
    
    print("🔍 Analyzing sssalud CAPTCHA patterns...")
    print("=" * 50)
    
    for captcha in real_captchas:
        print(f"CAPTCHA: {captcha}")
        for i, char in enumerate(captcha):
            total_chars += 1
            if char.isupper():
                uppercase_count += 1
                char_positions[i]['upper'] += 1
                print(f"  Position {i}: {char} (UPPERCASE)")
            elif char.islower():
                lowercase_count += 1
                char_positions[i]['lower'] += 1
                print(f"  Position {i}: {char} (lowercase)")
            elif char.isdigit():
                digit_count += 1
                char_positions[i]['digit'] += 1
                print(f"  Position {i}: {char} (DIGIT)")
        print()
    
    print("📊 OVERALL STATISTICS:")
    print(f"Total characters: {total_chars}")
    print(f"Uppercase: {uppercase_count} ({uppercase_count/total_chars*100:.1f}%)")
    print(f"Lowercase: {lowercase_count} ({lowercase_count/total_chars*100:.1f}%)")
    print(f"Digits: {digit_count} ({digit_count/total_chars*100:.1f}%)")
    
    print("\n📍 POSITION-WISE ANALYSIS:")
    for pos in range(6):
        total_pos = char_positions[pos]['upper'] + char_positions[pos]['lower'] + char_positions[pos]['digit']
        if total_pos > 0:
            upper_pct = char_positions[pos]['upper'] / total_pos * 100
            lower_pct = char_positions[pos]['lower'] / total_pos * 100
            digit_pct = char_positions[pos]['digit'] / total_pos * 100
            print(f"Position {pos}: Upper {upper_pct:.0f}%, Lower {lower_pct:.0f}%, Digits {digit_pct:.0f}%")
    
    print("\n💡 RECOMMENDATIONS:")
    print("1. Create training data with similar character distribution")
    print("2. Use fonts/styles that match sssalud appearance")
    print("3. Consider transfer learning from current model")
    print("4. Generate more varied training examples")

if __name__ == "__main__":
    analyze_captcha_patterns()
