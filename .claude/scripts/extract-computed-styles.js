/**
 * Extract Computed Styles from DOM
 *
 * This script extracts real CSS computed styles from a webpage's DOM
 * to provide accurate design tokens for UI replication.
 *
 * Usage: Execute this function via Chrome DevTools evaluate_script
 */

(() => {
  /**
   * Extract unique values from a set and sort them
   */
  const uniqueSorted = (set) => {
    return Array.from(set)
      .filter(v => v && v !== 'none' && v !== '0px' && v !== 'rgba(0, 0, 0, 0)')
      .sort();
  };

  /**
   * Parse rgb/rgba to OKLCH format (placeholder - returns original for now)
   */
  const toOKLCH = (color) => {
    // TODO: Implement actual RGB to OKLCH conversion
    // For now, return the original color with a note
    return `${color} /* TODO: Convert to OKLCH */`;
  };

  /**
   * Extract only key styles from an element
   */
  const extractKeyStyles = (element) => {
    const s = window.getComputedStyle(element);
    return {
      color: s.color,
      bg: s.backgroundColor,
      borderRadius: s.borderRadius,
      boxShadow: s.boxShadow,
      fontSize: s.fontSize,
      fontWeight: s.fontWeight,
      padding: s.padding,
      margin: s.margin
    };
  };

  /**
   * Main extraction function - extract all critical design tokens
   */
  const extractDesignTokens = () => {
    // Include all key UI elements
    const selectors = [
      'button', '.btn', '[role="button"]',
      'input', 'textarea', 'select',
      'h1', 'h2', 'h3', 'h4', 'h5', 'h6',
      '.card', 'article', 'section',
      'a', 'p', 'nav', 'header', 'footer'
    ];

    // Collect all design tokens
    const tokens = {
      colors: new Set(),
      borderRadii: new Set(),
      shadows: new Set(),
      fontSizes: new Set(),
      fontWeights: new Set(),
      spacing: new Set()
    };

    // Extract from all elements
    selectors.forEach(selector => {
      try {
        const elements = document.querySelectorAll(selector);
        elements.forEach(element => {
          const s = extractKeyStyles(element);

          // Collect all tokens (no limits)
          if (s.color && s.color !== 'rgba(0, 0, 0, 0)') tokens.colors.add(s.color);
          if (s.bg && s.bg !== 'rgba(0, 0, 0, 0)') tokens.colors.add(s.bg);
          if (s.borderRadius && s.borderRadius !== '0px') tokens.borderRadii.add(s.borderRadius);
          if (s.boxShadow && s.boxShadow !== 'none') tokens.shadows.add(s.boxShadow);
          if (s.fontSize) tokens.fontSizes.add(s.fontSize);
          if (s.fontWeight) tokens.fontWeights.add(s.fontWeight);

          // Extract all spacing values
          [s.padding, s.margin].forEach(val => {
            if (val && val !== '0px') {
              val.split(' ').forEach(v => {
                if (v && v !== '0px') tokens.spacing.add(v);
              });
            }
          });
        });
      } catch (e) {
        console.warn(`Error: ${selector}`, e);
      }
    });

    // Return all tokens (no element details to save context)
    return {
      metadata: {
        extractedAt: new Date().toISOString(),
        url: window.location.href,
        method: 'computed-styles'
      },
      tokens: {
        colors: uniqueSorted(tokens.colors),
        borderRadii: uniqueSorted(tokens.borderRadii),  // ALL radius values
        shadows: uniqueSorted(tokens.shadows),          // ALL shadows
        fontSizes: uniqueSorted(tokens.fontSizes),
        fontWeights: uniqueSorted(tokens.fontWeights),
        spacing: uniqueSorted(tokens.spacing)
      }
    };
  };

  // Execute and return results
  return extractDesignTokens();
})();
