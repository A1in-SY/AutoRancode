// ==UserScript==
// @name         AutoRancode
// @version      v1.0
// @author       A1in
// @namespace    https://github.com/A1in-SY
// @description  基于tensorflow.js实现自动填写华南师范大学sso系统验证码。
// @match        https://sso.scnu.edu.cn/AccountService/openapi/login.html*
// @match        https://sso.scnu.edu.cn/AccountService/user/login.html*
// @match        https://sso.scnu.edu.cn/AccountService/user/index.html
// @run-at       document-end
// @require      https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@2.0.0/dist/tf.min.js
// ==/UserScript==

const characters = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
const DBname = "autorancode"
let model;

function decode(y) {
    let code = '';
    //y.print();
    let z = tf.tensor(y.arraySync()[0], [4, 62]);
    //z.print();
    let t = tf.argMax(z, 1).arraySync();
    code = characters[t[0]] + characters[t[1]] + characters[t[2]] + characters[t[3]]
    return code;
}

async function run() {
    // 获取验证码图片，判断是否加载成功
    const imgEl = document.getElementById("codeimg");
    if (!imgEl) {
        return;
    }

    // 点击新验证码时重新识别
    imgEl.addEventListener("load", run);

    if (imgEl.height != 48 || imgEl.width != 96) {
        return
    }

    await indexedDB.open(DBname);
    // 打开模型，失败则退出
    try {
        model = await tf.loadGraphModel("indexeddb://" + DBname);
        console.log("从indexeddb加载模型成功");
    } catch (e) {
        console.log("从indexeddb加载模型失败，尝试从CDN加载模型");
        try {
            model = await tf.loadGraphModel("https://cdn.jsdelivr.net/gh/A1in-SY/AutoRancode@master/model_resnet_100_web/model.json");
            await model.save("indexeddb://" + DBname);
        } catch (e) {
            console.log("从CDN加载模型失败");
        }
    }

    if (!model) {
        console.log("没有可用的模型，退出");
        return;
    }

    //将验证码图片转换成模型的输入向量[1,1,50,100]
    let img = tf.browser.fromPixels(imgEl).mean(2).toFloat().expandDims(0).expandDims(-1);
    img = tf.image.resizeBilinear(img, [50, 100]);// 下载下来的样本是100*50,在这里得到的tensor是96*48
    //img = img.div(tf.scalar(255));
    img = img.reshape([1, 1, 50, 100]);

    const ret = model.execute(img);
    const preCode = decode(ret);

    // 填入验证码
    let rancodeEl = document.getElementById("rancode");
    rancodeEl.value = preCode;
}

(async function () {
    'use strict';

    window.addEventListener('load', run, false);
})();