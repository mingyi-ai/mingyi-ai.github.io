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

const field = document.querySelector("#dots-field");

if (field) {
    const viewport = { width: 0, height: 0, x: 0, y: 0 };
    const dots = [];
    const circle = { radius: 3, margin: 20 };
    const pointer = {
        x: 0,
        y: 0,
        previousX: 0,
        previousY: 0,
        speed: 0,
        initialized: false
    };

    function rebuildDots() {
        const bounds = field.getBoundingClientRect();
        viewport.width = bounds.width;
        viewport.height = bounds.height;
        viewport.x = bounds.left;
        viewport.y = bounds.top;

        field.replaceChildren();
        dots.length = 0;

        const spacing = circle.radius + circle.margin;
        const rows = Math.floor(viewport.height / spacing);
        const columns = Math.floor(viewport.width / spacing);
        const offsetX = (viewport.width % spacing) / 2;
        const offsetY = (viewport.height % spacing) / 2;

        for (let row = 0; row < rows; row += 1) {
            for (let column = 0; column < columns; column += 1) {
                const anchor = {
                    x: offsetX + column * spacing + spacing / 2,
                    y: offsetY + row * spacing + spacing / 2
                };
                const element = document.createElementNS(
                    "http://www.w3.org/2000/svg",
                    "circle"
                );

                element.setAttribute("cx", anchor.x);
                element.setAttribute("cy", anchor.y);
                element.setAttribute("r", circle.radius / 2);
                field.append(element);

                dots.push({
                    anchor,
                    position: { ...anchor },
                    smooth: { ...anchor },
                    velocity: { x: 0, y: 0 },
                    element
                });
            }
        }
    }

    let resizeFrame;
    window.addEventListener("resize", function () {
        cancelAnimationFrame(resizeFrame);
        resizeFrame = requestAnimationFrame(rebuildDots);
    });

    rebuildDots();

    if (!window.matchMedia("(prefers-reduced-motion: reduce)").matches) {
        window.addEventListener(
            "pointermove",
            function (event) {
                pointer.x = event.clientX;
                pointer.y = event.clientY;

                if (!pointer.initialized) {
                    pointer.previousX = pointer.x;
                    pointer.previousY = pointer.y;
                    pointer.initialized = true;
                }
            },
            { passive: true }
        );

        function animate() {
            const movement = Math.hypot(
                pointer.previousX - pointer.x,
                pointer.previousY - pointer.y
            );
            pointer.speed += (movement - pointer.speed) * 0.5;
            pointer.previousX = pointer.x;
            pointer.previousY = pointer.y;

            if (pointer.speed < 0.001) {
                pointer.speed = 0;
            }

            for (const dot of dots) {
                const distanceX =
                    pointer.x - viewport.x - dot.position.x;
                const distanceY =
                    pointer.y - viewport.y - dot.position.y;
                const distance = Math.max(
                    Math.hypot(distanceX, distanceY),
                    1
                );

                if (pointer.initialized && distance < 100) {
                    const angle = Math.atan2(distanceY, distanceX);
                    const movementAmount =
                        (500 / distance) * (pointer.speed * 0.1);
                    dot.velocity.x -= Math.cos(angle) * movementAmount;
                    dot.velocity.y -= Math.sin(angle) * movementAmount;
                }

                dot.velocity.x *= 0.9;
                dot.velocity.y *= 0.9;
                dot.position.x = dot.anchor.x + dot.velocity.x;
                dot.position.y = dot.anchor.y + dot.velocity.y;
                dot.smooth.x += (dot.position.x - dot.smooth.x) * 0.1;
                dot.smooth.y += (dot.position.y - dot.smooth.y) * 0.1;
                dot.element.setAttribute("cx", dot.smooth.x);
                dot.element.setAttribute("cy", dot.smooth.y);
            }

            requestAnimationFrame(animate);
        }

        requestAnimationFrame(animate);
    }
}
