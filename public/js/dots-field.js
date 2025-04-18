/*
 * Dots Field Animation
 * Original code by Antoine Wodniack (https://codepen.io/wodniack/pen/abWNWGW)
 * Licensed under the MIT License
 * Copyright (c) 2025 Antoine Wodniack
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */

/* Properties */
const svg = {
    el: document.querySelector('#dots-field'),
    width: 1,
    height: 1,
    x: 0,
    y: 0
  };
  
  const dots = [];
  
  const circle = {
    radius: 3,
    margin: 20
  };
  
  const mouse = {
    x: 0,
    y: 0,
    prevX: 0,
    prevY: 0,
    speed: 0
  };
  
  /* Resize */
  function resizeHandler() {
    const bounding = svg.el.getBoundingClientRect();
  
    svg.width = bounding.width;
    svg.height = bounding.height;
    svg.x = bounding.left;
    svg.y = bounding.top;
  }
  
  /* Create dots */
  function createDots() {
    resizeHandler();
  
    const dotSize = circle.radius + circle.margin;
  
    const rows = Math.floor(svg.height / dotSize);
    const cols = Math.floor(svg.width / dotSize);
  
    const x = (svg.width % dotSize) / 2;
    const y = (svg.height % dotSize) / 2;
  
    for (let row = 0; row < rows; row++) {
      for (let col = 0; col < cols; col++) {
        const dot = {
          anchor: {
            x: x + col * dotSize + dotSize / 2,
            y: y + row * dotSize + dotSize / 2
          }
        };
  
        dot.position = { x: dot.anchor.x, y: dot.anchor.y };
        dot.smooth = { x: dot.anchor.x, y: dot.anchor.y };
        dot.velocity = { x: 0, y: 0 };
  
        dot.el = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
        dot.el.setAttribute('cx', dot.anchor.x);
        dot.el.setAttribute('cy', dot.anchor.y);
        dot.el.setAttribute('r', circle.radius / 2);
  
        svg.el.append(dot.el);
        dots.push(dot);
      }
    }
  }
  
  /* Check mouse move */
  function mouseHandler(e) {
    mouse.x = e.pageX - window.scrollX;
    mouse.y = e.pageY - window.scrollY;
  }
  
  /* Check mouse speed */
  function mouseSpeed() {
    const distX = mouse.prevX - mouse.x;
    const distY = mouse.prevY - mouse.y;
    const dist = Math.hypot(distX, distY);
  
    mouse.speed += (dist - mouse.speed) * 0.5;
    if (mouse.speed < 0.001) {
      mouse.speed = 0;
    }
  
    mouse.prevX = mouse.x;
    mouse.prevY = mouse.y;
  
    setTimeout(mouseSpeed, 20);
  }
  
  /* Tick */
  function tick() {
    dots.forEach((dot) => {
      const distX = mouse.x - svg.x - dot.position.x;
      const distY = mouse.y - svg.y - dot.position.y;
      const dist = Math.max(Math.hypot(distX, distY), 1);
  
      const angle = Math.atan2(distY, distX);
  
      const move = (500 / dist) * (mouse.speed * 0.1);
  
      if (dist < 100) {
        dot.velocity.x += Math.cos(angle) * -move;
        dot.velocity.y += Math.sin(angle) * -move;
      }
  
      dot.velocity.x *= 0.9;
      dot.velocity.y *= 0.9;
  
      dot.position.x = dot.anchor.x + dot.velocity.x;
      dot.position.y = dot.anchor.y + dot.velocity.y;
  
      dot.smooth.x += (dot.position.x - dot.smooth.x) * 0.1;
      dot.smooth.y += (dot.position.y - dot.smooth.y) * 0.1;
  
      dot.el.setAttribute('cx', dot.smooth.x);
      dot.el.setAttribute('cy', dot.smooth.y);
    });
  
    requestAnimationFrame(tick);
  }
  
  /* Ready */
  (function () {
    // Resize
    window.addEventListener('resize', resizeHandler);
  
    // Mouse
    window.addEventListener('mousemove', mouseHandler);
    mouseSpeed();
  
    // Dots
    createDots();
  
    // Tick
    tick();
  })();