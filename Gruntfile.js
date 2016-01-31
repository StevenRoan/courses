const exec = require('child_process').exec;
const spawn = require('child_process').spawn;
var util = require('util');

module.exports = function (grunt) {
    grunt.registerTask("python", "run python", function (done) {
        var done = this.async();
        var filename = "q11.py"
        exec(util.format("cd ./machine-learning-NTU-2015/techniques/hw4 && python", filename), function (err, stdout, stderr) {
            console.log("------------------------------------")
            if (err) {
                // console.log(err);
            }
            if (stdout) console.log(stdout)
            if (stderr) console.error(stderr);
            console.log("------------------------------------")
            // done()
        });
    });

    // Project configuration.
    grunt.initConfig({
        watch: {
            scripts: {
                files: ['machine-learning-NTU-2015/techniques/hw4/*.py', 'machine-learning-NTU-2015/techniques/hw4/examples/*.py','*.js', 'machine-learning-NTU-2015/techniques/hw4/lib/*.py'],
                tasks: ["python"],
                options: {
                    interrupt: true
                }
            }
        }
    });

    // Load the plugin that provides the "uglify" task.
    grunt.loadNpmTasks('grunt-contrib-watch');

};
