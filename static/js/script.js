
document.addEventListener("DOMContentLoaded", function() {
    gsap.from(".hero-content", { duration: 2, y: 50, opacity: 0, ease: "power2.out" });

    gsap.from(".about-section h2", {
        scrollTrigger: {
            trigger: ".about-section",
            start: "top 80%",
            toggleActions: "play none none none"
        },
        duration: 1.2,
        y: 50,
        opacity: 0,
        ease: "power2.out"
    });

    gsap.from(".about-section p", {
        scrollTrigger: {
            trigger: ".about-section",
            start: "top 75%",
            toggleActions: "play none none none"
        },
        duration: 1.5,
        y: 50,
        opacity: 0,
        ease: "power2.out",
        delay: 0.3
    });

    // Smooth scrolling for navigation links
    document.querySelectorAll("nav a, .btn").forEach(anchor => {
        anchor.addEventListener("click", function(e) {
            e.preventDefault();
            const targetId = this.getAttribute("href").substring(1);
            document.getElementById(targetId).scrollIntoView({ behavior: "smooth" });
        });
    });
});





        document.querySelector('.menu-toggle').addEventListener('click', function() {
            document.querySelector('nav ul').classList.toggle('active');
        });




    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener("click", function (e) {
            e.preventDefault();
            document.querySelector(this.getAttribute("href")).scrollIntoView({
                behavior: "smooth"
            });
        });
    });













