/**
 * Animation & Transition Extraction Script
 *
 * Extracts CSS animations, transitions, and transform patterns from a live web page.
 * This script runs in the browser context via Chrome DevTools Protocol.
 *
 * @returns {Object} Structured animation data
 */
(() => {
  const extractionTimestamp = new Date().toISOString();
  const currentUrl = window.location.href;

  /**
   * Parse transition shorthand or individual properties
   */
  function parseTransition(element, computedStyle) {
    const transition = computedStyle.transition || computedStyle.webkitTransition;

    if (!transition || transition === 'none' || transition === 'all 0s ease 0s') {
      return null;
    }

    // Parse shorthand: "property duration easing delay"
    const transitions = [];
    const parts = transition.split(/,\s*/);

    parts.forEach(part => {
      const match = part.match(/^(\S+)\s+([\d.]+m?s)\s+(\S+)(?:\s+([\d.]+m?s))?/);
      if (match) {
        transitions.push({
          property: match[1],
          duration: match[2],
          easing: match[3],
          delay: match[4] || '0s'
        });
      }
    });

    return transitions.length > 0 ? transitions : null;
  }

  /**
   * Extract animation name and properties
   */
  function parseAnimation(element, computedStyle) {
    const animationName = computedStyle.animationName || computedStyle.webkitAnimationName;

    if (!animationName || animationName === 'none') {
      return null;
    }

    return {
      name: animationName,
      duration: computedStyle.animationDuration || computedStyle.webkitAnimationDuration,
      easing: computedStyle.animationTimingFunction || computedStyle.webkitAnimationTimingFunction,
      delay: computedStyle.animationDelay || computedStyle.webkitAnimationDelay || '0s',
      iterationCount: computedStyle.animationIterationCount || computedStyle.webkitAnimationIterationCount || '1',
      direction: computedStyle.animationDirection || computedStyle.webkitAnimationDirection || 'normal',
      fillMode: computedStyle.animationFillMode || computedStyle.webkitAnimationFillMode || 'none'
    };
  }

  /**
   * Extract transform value
   */
  function parseTransform(computedStyle) {
    const transform = computedStyle.transform || computedStyle.webkitTransform;

    if (!transform || transform === 'none') {
      return null;
    }

    return transform;
  }

  /**
   * Get element selector (simplified for readability)
   */
  function getSelector(element) {
    if (element.id) {
      return `#${element.id}`;
    }

    if (element.className && typeof element.className === 'string') {
      const classes = element.className.trim().split(/\s+/).slice(0, 2).join('.');
      if (classes) {
        return `.${classes}`;
      }
    }

    return element.tagName.toLowerCase();
  }

  /**
   * Extract all stylesheets and find @keyframes rules
   */
  function extractKeyframes() {
    const keyframes = {};

    try {
      // Iterate through all stylesheets
      Array.from(document.styleSheets).forEach(sheet => {
        try {
          // Skip external stylesheets due to CORS
          if (sheet.href && !sheet.href.startsWith(window.location.origin)) {
            return;
          }

          Array.from(sheet.cssRules || sheet.rules || []).forEach(rule => {
            // Check for @keyframes rules
            if (rule.type === CSSRule.KEYFRAMES_RULE || rule.type === CSSRule.WEBKIT_KEYFRAMES_RULE) {
              const name = rule.name;
              const frames = {};

              Array.from(rule.cssRules || []).forEach(keyframe => {
                const key = keyframe.keyText; // e.g., "0%", "50%", "100%"
                frames[key] = keyframe.style.cssText;
              });

              keyframes[name] = frames;
            }
          });
        } catch (e) {
          // Skip stylesheets that can't be accessed (CORS)
          console.warn('Cannot access stylesheet:', sheet.href, e.message);
        }
      });
    } catch (e) {
      console.error('Error extracting keyframes:', e);
    }

    return keyframes;
  }

  /**
   * Scan visible elements for animations and transitions
   */
  function scanElements() {
    const elements = document.querySelectorAll('*');
    const transitionData = [];
    const animationData = [];
    const transformData = [];

    const uniqueTransitions = new Set();
    const uniqueAnimations = new Set();
    const uniqueEasings = new Set();
    const uniqueDurations = new Set();

    elements.forEach(element => {
      // Skip invisible elements
      const rect = element.getBoundingClientRect();
      if (rect.width === 0 && rect.height === 0) {
        return;
      }

      const computedStyle = window.getComputedStyle(element);

      // Extract transitions
      const transitions = parseTransition(element, computedStyle);
      if (transitions) {
        const selector = getSelector(element);
        transitions.forEach(t => {
          const key = `${t.property}-${t.duration}-${t.easing}`;
          if (!uniqueTransitions.has(key)) {
            uniqueTransitions.add(key);
            transitionData.push({
              selector,
              ...t
            });
            uniqueEasings.add(t.easing);
            uniqueDurations.add(t.duration);
          }
        });
      }

      // Extract animations
      const animation = parseAnimation(element, computedStyle);
      if (animation) {
        const selector = getSelector(element);
        const key = `${animation.name}-${animation.duration}`;
        if (!uniqueAnimations.has(key)) {
          uniqueAnimations.add(key);
          animationData.push({
            selector,
            ...animation
          });
          uniqueEasings.add(animation.easing);
          uniqueDurations.add(animation.duration);
        }
      }

      // Extract transforms (on hover/active, we only get current state)
      const transform = parseTransform(computedStyle);
      if (transform) {
        const selector = getSelector(element);
        transformData.push({
          selector,
          transform
        });
      }
    });

    return {
      transitions: transitionData,
      animations: animationData,
      transforms: transformData,
      uniqueEasings: Array.from(uniqueEasings),
      uniqueDurations: Array.from(uniqueDurations)
    };
  }

  /**
   * Main extraction function
   */
  function extractAnimations() {
    const elementData = scanElements();
    const keyframes = extractKeyframes();

    return {
      metadata: {
        timestamp: extractionTimestamp,
        url: currentUrl,
        method: 'chrome-devtools',
        version: '1.0.0'
      },
      transitions: elementData.transitions,
      animations: elementData.animations,
      transforms: elementData.transforms,
      keyframes: keyframes,
      summary: {
        total_transitions: elementData.transitions.length,
        total_animations: elementData.animations.length,
        total_transforms: elementData.transforms.length,
        total_keyframes: Object.keys(keyframes).length,
        unique_easings: elementData.uniqueEasings,
        unique_durations: elementData.uniqueDurations
      }
    };
  }

  // Execute extraction
  return extractAnimations();
})();
