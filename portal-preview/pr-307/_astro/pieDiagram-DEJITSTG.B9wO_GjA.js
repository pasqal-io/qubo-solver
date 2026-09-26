import{t as e}from"./mermaid-parser.core.CSkM02_t.js";import{g as t,h as n}from"./src.Bgq8f4uP.js";import{B as r,C as i,V as a,W as o,_ as s,a as c,b as l,c as u,d,v as f}from"./chunk-ICPOFSXX.8srteSYA.js";import{t as p}from"./ordinal.BAxfPAgN.js";import{n as m}from"./path.fybaL0A-.js";import{m as h}from"./dist.DWYHC9IO.js";import{t as g}from"./arc.BZnNcxt1.js";import{t as _}from"./array.BifhSqXX.js";import{f as v,r as y}from"./chunk-5PVQY5BW.CPGvp4Pd.js";import{t as b}from"./chunk-4BX2VUAB.CihIbkmf.js";import{a as x}from"./mermaid.core.Btfvfn0U.js";function S(e,t){return t<e?-1:t>e?1:t>=e?0:NaN}function C(e){return e}function w(){var e=C,t=S,n=null,r=m(0),i=m(h),a=m(0);function o(o){var s,c=(o=_(o)).length,l,u,d=0,f=Array(c),p=Array(c),m=+r.apply(this,arguments),g=Math.min(h,Math.max(-h,i.apply(this,arguments)-m)),v,y=Math.min(Math.abs(g)/c,a.apply(this,arguments)),b=y*(g<0?-1:1),x;for(s=0;s<c;++s)(x=p[f[s]=s]=+e(o[s],s,o))>0&&(d+=x);for(t==null?n!=null&&f.sort(function(e,t){return n(o[e],o[t])}):f.sort(function(e,n){return t(p[e],p[n])}),s=0,u=d?(g-c*b)/d:0;s<c;++s,m=v)l=f[s],x=p[l],v=m+(x>0?x*u:0)+b,p[l]={data:o[l],index:s,value:x,startAngle:m,endAngle:v,padAngle:y};return p}return o.value=function(t){return arguments.length?(e=typeof t==`function`?t:m(+t),o):e},o.sortValues=function(e){return arguments.length?(t=e,n=null,o):t},o.sort=function(e){return arguments.length?(n=e,t=null,o):n},o.startAngle=function(e){return arguments.length?(r=typeof e==`function`?e:m(+e),o):r},o.endAngle=function(e){return arguments.length?(i=typeof e==`function`?e:m(+e),o):i},o.padAngle=function(e){return arguments.length?(a=typeof e==`function`?e:m(+e),o):a},o}var T=d.pie,E={sections:new Map,showData:!1,config:T},D=E.sections,O=E.showData,k=structuredClone(T),A={getConfig:n(()=>structuredClone(k),`getConfig`),clear:n(()=>{D=new Map,O=E.showData,c()},`clear`),setDiagramTitle:o,getDiagramTitle:i,setAccTitle:a,getAccTitle:f,setAccDescription:r,getAccDescription:s,addSection:n(({label:e,value:n})=>{if(n<0)throw Error(`"${e}" has invalid value: ${n}. Negative values are not allowed in pie charts. All slice values must be >= 0.`);D.has(e)||(D.set(e,n),t.debug(`added new section: ${e}, with value: ${n}`))},`addSection`),getSections:n(()=>D,`getSections`),setShowData:n(e=>{O=e},`setShowData`),getShowData:n(()=>O,`getShowData`)},j=n((e,t)=>{b(e,t),t.setShowData(e.showData),e.sections.map(t.addSection)},`populateDb`),M={parse:n(async n=>{let r=await e(`pie`,n);t.debug(r),j(r,A)},`parse`)},N=n(e=>`
  .pieCircle{
    stroke: ${e.pieStrokeColor};
    stroke-width : ${e.pieStrokeWidth};
    opacity : ${e.pieOpacity};
  }
  .pieOuterCircle{
    stroke: ${e.pieOuterStrokeColor};
    stroke-width: ${e.pieOuterStrokeWidth};
    fill: none;
  }
  .pieTitleText {
    text-anchor: middle;
    font-size: ${e.pieTitleTextSize};
    fill: ${e.pieTitleTextColor};
    font-family: ${e.fontFamily};
  }
  .slice {
    font-family: ${e.fontFamily};
    fill: ${e.pieSectionTextColor};
    font-size:${e.pieSectionTextSize};
    // fill: white;
  }
  .legend text {
    fill: ${e.pieLegendTextColor};
    font-family: ${e.fontFamily};
    font-size: ${e.pieLegendTextSize};
  }
`,`getStyles`),P=n(e=>{let t=[...e.values()].reduce((e,t)=>e+t,0),n=[...e.entries()].map(([e,t])=>({label:e,value:t})).filter(e=>e.value/t*100>=1);return w().value(e=>e.value).sort(null)(n)},`createPieArcs`),F={parser:M,db:A,renderer:{draw:n((e,n,r,i)=>{t.debug(`rendering pie chart
`+e);let a=i.db,o=l(),s=y(a.getConfig(),o.pie),c=x(n),d=c.append(`g`);d.attr(`transform`,`translate(225,225)`);let{themeVariables:f}=o,[m]=v(f.pieOuterStrokeWidth);m??=2;let h=s.textPosition,_=g().innerRadius(0).outerRadius(185),b=g().innerRadius(185*h).outerRadius(185*h);d.append(`circle`).attr(`cx`,0).attr(`cy`,0).attr(`r`,185+m/2).attr(`class`,`pieOuterCircle`);let S=a.getSections(),C=P(S),w=[f.pie1,f.pie2,f.pie3,f.pie4,f.pie5,f.pie6,f.pie7,f.pie8,f.pie9,f.pie10,f.pie11,f.pie12],T=0;S.forEach(e=>{T+=e});let E=C.filter(e=>(e.data.value/T*100).toFixed(0)!==`0`),D=p(w).domain([...S.keys()]);d.selectAll(`mySlices`).data(E).enter().append(`path`).attr(`d`,_).attr(`fill`,e=>D(e.data.label)).attr(`class`,`pieCircle`),d.selectAll(`mySlices`).data(E).enter().append(`text`).text(e=>(e.data.value/T*100).toFixed(0)+`%`).attr(`transform`,e=>`translate(`+b.centroid(e)+`)`).style(`text-anchor`,`middle`).attr(`class`,`slice`);let O=d.append(`text`).text(a.getDiagramTitle()).attr(`x`,0).attr(`y`,-200).attr(`class`,`pieTitleText`),k=[...S.entries()].map(([e,t])=>({label:e,value:t})),A=d.selectAll(`.legend`).data(k).enter().append(`g`).attr(`class`,`legend`).attr(`transform`,(e,t)=>{let n=22*k.length/2;return`translate(216,`+(t*22-n)+`)`});A.append(`rect`).attr(`width`,18).attr(`height`,18).style(`fill`,e=>D(e.label)).style(`stroke`,e=>D(e.label)),A.append(`text`).attr(`x`,22).attr(`y`,14).text(e=>a.getShowData()?`${e.label} [${e.value}]`:e.label);let j=512+Math.max(...A.selectAll(`text`).nodes().map(e=>e?.getBoundingClientRect().width??0)),M=O.node()?.getBoundingClientRect().width??0,N=225-M/2,F=225+M/2,I=Math.min(0,N),L=Math.max(j,F)-I;c.attr(`viewBox`,`${I} 0 ${L} 450`),u(c,450,L,s.useMaxWidth)},`draw`)},styles:N};export{F as diagram};