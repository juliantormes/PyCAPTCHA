#!/usr/bin/env python3
"""
Updated analysis with 20 real sssalud CAPTCHAs for better pattern recognition
"""

def analyze_extended_captcha_patterns():
    # Original 10 + New 10 (need real values for 11-20)
    real_captchas = [
        # Original 10
        "UKhGh9",  # 1.png
        "26WanS",  # 2.png
        "e4TkHP",  # 3.png
        "cGfFE2",  # 4.png
        "gnRYZe",  # 5.png
        "v76Ebu",  # 6.png
        "DUzp49",  # 7.png
        "MWR3mw",  # 8.png
        "3h2vUF",  # 9.png
        "t2md2m",  # 10.png
        
        # New 10 (real values)
        "CzSLcN",  # 11.png
        "tZyFmr",  # 12.png
        "3N4kv3",  # 13.png
        "wvTfzE",  # 14.png
        "z56VDI",  # 15.png
        "9mLpn5",  # 16.png
        "pKK7BD",  # 17.png
        "TpHfBC",  # 18.png
        "rpA8pS",  # 19.png
        "Z3TUnp",  # 20.png
    ]
    
    print("🔍 Analyzing extended sssalud CAPTCHA patterns (20 samples)...")
    print("=" * 60)
    
    # Character type counters
    total_chars = 0
    uppercase_count = 0
    lowercase_count = 0
    digit_count = 0
    
    # Position-wise analysis
    position_stats = {i: {'upper': 0, 'lower': 0, 'digit': 0} for i in range(6)}
    
    # Character frequency analysis
    char_frequency = {}
    
    for i, captcha in enumerate(real_captchas, 1):
        if captcha == "XXXXXX":
            print(f"⚠️  CAPTCHA {i}: PENDING - Need real value")
            continue
            
        print(f"CAPTCHA {i}: {captcha}")
        
        for pos, char in enumerate(captcha):
            total_chars += 1
            
            # Count character frequency
            char_frequency[char] = char_frequency.get(char, 0) + 1
            
            if char.isupper():
                char_type = "UPPERCASE"
                uppercase_count += 1
                position_stats[pos]['upper'] += 1
            elif char.islower():
                char_type = "lowercase"
                lowercase_count += 1
                position_stats[pos]['lower'] += 1
            elif char.isdigit():
                char_type = "DIGIT"
                digit_count += 1
                position_stats[pos]['digit'] += 1
            
            print(f"  Position {pos}: {char} ({char_type})")
    
    # Calculate percentages
    if total_chars > 0:
        print(f"\n📊 OVERALL STATISTICS ({total_chars} characters):")
        print(f"Uppercase: {uppercase_count} ({uppercase_count/total_chars*100:.1f}%)")
        print(f"Lowercase: {lowercase_count} ({lowercase_count/total_chars*100:.1f}%)")
        print(f"Digits: {digit_count} ({digit_count/total_chars*100:.1f}%)")
        
        print(f"\n📍 POSITION-WISE ANALYSIS:")
        valid_samples = sum(1 for c in real_captchas if c != "XXXXXX")
        
        for pos in range(6):
            if valid_samples > 0:
                upper_pct = position_stats[pos]['upper'] / valid_samples * 100
                lower_pct = position_stats[pos]['lower'] / valid_samples * 100
                digit_pct = position_stats[pos]['digit'] / valid_samples * 100
                print(f"Position {pos}: Upper {upper_pct:.0f}%, Lower {lower_pct:.0f}%, Digits {digit_pct:.0f}%")
        
        print(f"\n🔤 CHARACTER FREQUENCY (Top 15):")
        sorted_chars = sorted(char_frequency.items(), key=lambda x: x[1], reverse=True)
        for char, count in sorted_chars[:15]:
            pct = count / total_chars * 100
            print(f"  '{char}': {count} times ({pct:.1f}%)")
        
        print(f"\n💡 UPDATED RECOMMENDATIONS:")
        print("1. ✅ Better character distribution with 20 samples")
        print("2. 🎯 More accurate position-wise patterns")
        print("3. 📈 Improved generator training data")
        print("4. 🔍 Better coverage of edge cases")
        
        # Generate improved character sets for the generator
        print(f"\n🛠️ GENERATOR IMPROVEMENTS:")
        all_chars = set(''.join(real_captchas).replace('X', ''))
        print(f"Total unique characters found: {len(all_chars)}")
        print(f"Characters: {sorted(all_chars)}")
        
        # Position-specific character sets
        print(f"\n📍 POSITION-SPECIFIC CHARACTER SETS:")
        for pos in range(6):
            pos_chars = set()
            for captcha in real_captchas:
                if captcha != "XXXXXX" and len(captcha) > pos:
                    pos_chars.add(captcha[pos])
            print(f"Position {pos}: {sorted(pos_chars)} ({len(pos_chars)} unique)")

if __name__ == "__main__":
    analyze_extended_captcha_patterns()
