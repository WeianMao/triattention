/**
 * Extract Layout Structure from DOM - Enhanced Version
 *
 * Extracts real layout information from DOM to provide accurate
 * structural data for UI replication.
 *
 * Features:
 * - Framework detection (Nuxt.js, Next.js, React, Vue, Angular)
 * - Multi-strategy container detection (strict → relaxed → class-based → framework-specific)
 * - Intelligent main content detection with common class names support
 * - Supports modern SPA frameworks
 * - Detects non-semantic main containers (.main, .content, etc.)
 * - Progressive exploration: Auto-discovers missing selectors when standard patterns fail
 * - Suggests new class names to add to script based on actual page structure
 *
 * Progressive Exploration:
 * When fewer than 3 main containers are found, the script automatically:
 * 1. Analyzes all large visible containers (≥500×300px)
 * 2. Extracts class name patterns (main/content/wrapper/container/page/etc.)
 * 3. Suggests new selectors to add to the script
 * 4. Returns exploration data in result.exploration
 *
 * Usage: Execute via Chrome DevTools evaluate_script
 * Version: 2.2.0
 */

(() => {
  /**
   * Get element's bounding box relative to viewport
   */
  const getBounds = (element) => {
    const rect = element.getBoundingClientRect();
    return {
      x: Math.round(rect.x),
      y: Math.round(rect.y),
      width: Math.round(rect.width),
      height: Math.round(rect.height)
    };
  };

  /**
   * Extract layout properties from an element
   */
  const extractLayoutProps = (element) => {
    const s = window.getComputedStyle(element);

    return {
      // Core layout
      display: s.display,
      position: s.position,

      // Flexbox
      flexDirection: s.flexDirection,
      justifyContent: s.justifyContent,
      alignItems: s.alignItems,
      flexWrap: s.flexWrap,
      gap: s.gap,

      // Grid
      gridTemplateColumns: s.gridTemplateColumns,
      gridTemplateRows: s.gridTemplateRows,
      gridAutoFlow: s.gridAutoFlow,

      // Dimensions
      width: s.width,
      height: s.height,
      maxWidth: s.maxWidth,
      minWidth: s.minWidth,

      // Spacing
      padding: s.padding,
      margin: s.margin
    };
  };

  /**
   * Identify layout pattern for an element
   */
  const identifyPattern = (props) => {
    const { display, flexDirection, gridTemplateColumns } = props;

    if (display === 'flex' || display === 'inline-flex') {
      if (flexDirection === 'column') return 'flex-column';
      if (flexDirection === 'row') return 'flex-row';
      return 'flex';
    }

    if (display === 'grid') {
      const cols = gridTemplateColumns;
      if (cols && cols !== 'none') {
        const colCount = cols.split(' ').length;
        return `grid-${colCount}col`;
      }
      return 'grid';
    }

    if (display === 'block') return 'block';

    return display;
  };

  /**
   * Detect frontend framework
   */
  const detectFramework = () => {
    if (document.querySelector('#__nuxt')) return { name: 'Nuxt.js', version: 'unknown' };
    if (document.querySelector('#__next')) return { name: 'Next.js', version: 'unknown' };
    if (document.querySelector('[data-reactroot]')) return { name: 'React', version: 'unknown' };
    if (document.querySelector('[ng-version]')) return { name: 'Angular', version: 'unknown' };
    if (window.Vue) return { name: 'Vue.js', version: window.Vue.version || 'unknown' };
    return { name: 'Unknown', version: 'unknown' };
  };

  /**
   * Build layout tree recursively
   */
  const buildLayoutTree = (element, depth = 0, maxDepth = 3) => {
    if (depth > maxDepth) return null;

    const props = extractLayoutProps(element);
    const bounds = getBounds(element);
    const pattern = identifyPattern(props);

    // Get semantic role
    const tagName = element.tagName.toLowerCase();
    const classes = Array.from(element.classList).slice(0, 3); // Max 3 classes
    const role = element.getAttribute('role');

    // Build node
    const node = {
      tag: tagName,
      classes: classes,
      role: role,
      pattern: pattern,
      bounds: bounds,
      layout: {
        display: props.display,
        position: props.position
      }
    };

    // Add flex/grid specific properties
    if (props.display === 'flex' || props.display === 'inline-flex') {
      node.layout.flexDirection = props.flexDirection;
      node.layout.justifyContent = props.justifyContent;
      node.layout.alignItems = props.alignItems;
      node.layout.gap = props.gap;
    }

    if (props.display === 'grid') {
      node.layout.gridTemplateColumns = props.gridTemplateColumns;
      node.layout.gridTemplateRows = props.gridTemplateRows;
      node.layout.gap = props.gap;
    }

    // Process children for container elements
    if (props.display === 'flex' || props.display === 'grid' || props.display === 'block') {
      const children = Array.from(element.children);
      if (children.length > 0 && children.length < 50) { // Limit to 50 children
        node.children = children
          .map(child => buildLayoutTree(child, depth + 1, maxDepth))
          .filter(child => child !== null);
      }
    }

    return node;
  };

  /**
   * Find main layout containers with multi-strategy approach
   */
  const findMainContainers = () => {
    const containers = [];
    const found = new Set();

    // Strategy 1: Strict selectors (body direct children)
    const strictSelectors = [
      'body > header',
      'body > nav',
      'body > main',
      'body > footer'
    ];

    // Strategy 2: Relaxed selectors (any level)
    const relaxedSelectors = [
      'header',
      'nav',
      'main',
      'footer',
      '[role="banner"]',
      '[role="navigation"]',
      '[role="main"]',
      '[role="contentinfo"]'
    ];

    // Strategy 3: Common class-based main content selectors
    const commonClassSelectors = [
      '.main',
      '.content',
      '.main-content',
      '.page-content',
      '.container.main',
      '.wrapper > .main',
      'div[class*="main-wrapper"]',
      'div[class*="content-wrapper"]'
    ];

    // Strategy 4: Framework-specific selectors
    const frameworkSelectors = [
      '#__nuxt header', '#__nuxt .main', '#__nuxt main', '#__nuxt footer',
      '#__next header', '#__next .main', '#__next main', '#__next footer',
      '#app header', '#app .main', '#app main', '#app footer',
      '[data-app] header', '[data-app] .main', '[data-app] main', '[data-app] footer'
    ];

    // Try all strategies
    const allSelectors = [...strictSelectors, ...relaxedSelectors, ...commonClassSelectors, ...frameworkSelectors];

    allSelectors.forEach(selector => {
      try {
        const elements = document.querySelectorAll(selector);
        elements.forEach(element => {
          // Avoid duplicates and invisible elements
          if (!found.has(element) && element.offsetParent !== null) {
            found.add(element);
            const tree = buildLayoutTree(element, 0, 3);
            if (tree && tree.bounds.width > 0 && tree.bounds.height > 0) {
              containers.push(tree);
            }
          }
        });
      } catch (e) {
        console.warn(`Selector failed: ${selector}`, e);
      }
    });

    // Fallback: If no containers found, use body's direct children
    if (containers.length === 0) {
      Array.from(document.body.children).forEach(child => {
        if (child.offsetParent !== null && !found.has(child)) {
          const tree = buildLayoutTree(child, 0, 2);
          if (tree && tree.bounds.width > 100 && tree.bounds.height > 100) {
            containers.push(tree);
          }
        }
      });
    }

    return containers;
  };

  /**
   * Progressive exploration: Discover main containers when standard selectors fail
   * Analyzes large visible containers and suggests class name patterns
   */
  const exploreMainContainers = () => {
    const candidates = [];
    const minWidth = 500;
    const minHeight = 300;

    // Find all large visible divs
    const allDivs = document.querySelectorAll('div');
    allDivs.forEach(div => {
      const rect = div.getBoundingClientRect();
      const style = window.getComputedStyle(div);

      // Filter: large size, visible, not header/footer
      if (rect.width >= minWidth &&
          rect.height >= minHeight &&
          div.offsetParent !== null &&
          !div.closest('header') &&
          !div.closest('footer')) {

        const classes = Array.from(div.classList);
        const area = rect.width * rect.height;

        candidates.push({
          element: div,
          classes: classes,
          area: area,
          bounds: {
            width: Math.round(rect.width),
            height: Math.round(rect.height)
          },
          display: style.display,
          depth: getElementDepth(div)
        });
      }
    });

    // Sort by area (largest first) and take top candidates
    candidates.sort((a, b) => b.area - a.area);

    // Extract unique class patterns from top candidates
    const classPatterns = new Set();
    candidates.slice(0, 20).forEach(c => {
      c.classes.forEach(cls => {
        // Identify potential main content class patterns
        if (cls.match(/main|content|container|wrapper|page|body|layout|app/i)) {
          classPatterns.add(cls);
        }
      });
    });

    return {
      candidates: candidates.slice(0, 10).map(c => ({
        classes: c.classes,
        bounds: c.bounds,
        display: c.display,
        depth: c.depth
      })),
      suggestedSelectors: Array.from(classPatterns).map(cls => `.${cls}`)
    };
  };

  /**
   * Get element depth in DOM tree
   */
  const getElementDepth = (element) => {
    let depth = 0;
    let current = element;
    while (current.parentElement) {
      depth++;
      current = current.parentElement;
    }
    return depth;
  };

  /**
   * Analyze layout patterns
   */
  const analyzePatterns = (containers) => {
    const patterns = {
      flexColumn: 0,
      flexRow: 0,
      grid: 0,
      sticky: 0,
      fixed: 0
    };

    const analyze = (node) => {
      if (!node) return;

      if (node.pattern === 'flex-column') patterns.flexColumn++;
      if (node.pattern === 'flex-row') patterns.flexRow++;
      if (node.pattern && node.pattern.startsWith('grid')) patterns.grid++;
      if (node.layout.position === 'sticky') patterns.sticky++;
      if (node.layout.position === 'fixed') patterns.fixed++;

      if (node.children) {
        node.children.forEach(analyze);
      }
    };

    containers.forEach(analyze);
    return patterns;
  };

  /**
   * Main extraction function with progressive exploration
   */
  const extractLayout = () => {
    const framework = detectFramework();
    const containers = findMainContainers();
    const patterns = analyzePatterns(containers);

    // Progressive exploration: if too few containers found, explore and suggest
    let exploration = null;
    const minExpectedContainers = 3; // At least header, main, footer

    if (containers.length < minExpectedContainers) {
      exploration = exploreMainContainers();

      // Add warning message
      exploration.warning = `Only ${containers.length} containers found. Consider adding these selectors to the script:`;
      exploration.recommendation = exploration.suggestedSelectors.join(', ');
    }

    const result = {
      metadata: {
        extractedAt: new Date().toISOString(),
        url: window.location.href,
        framework: framework,
        method: 'layout-structure-enhanced',
        version: '2.2.0'
      },
      statistics: {
        totalContainers: containers.length,
        patterns: patterns
      },
      structure: containers
    };

    // Add exploration results if triggered
    if (exploration) {
      result.exploration = {
        triggered: true,
        reason: 'Insufficient containers found with standard selectors',
        discoveredCandidates: exploration.candidates,
        suggestedSelectors: exploration.suggestedSelectors,
        warning: exploration.warning,
        recommendation: exploration.recommendation
      };
    }

    return result;
  };

  // Execute and return results
  return extractLayout();
})();
