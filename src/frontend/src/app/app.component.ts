import { Component, OnInit } from '@angular/core';
import { HttpService } from './http.service';
import { CommonService } from './common.service';

@Component({
  selector: 'app-root',
  templateUrl: './app.component.html',
  styles: [
    "button.btn-action { width: 65px; }"
  ],
})
export class AppComponent implements OnInit {

  scrambleDepth = 10;

  constructor(public commonService: CommonService, public httpService: HttpService) { }

  ngOnInit() {
    this.getInitData();
  }

  private async getInitData() {
    await Promise.all([
      this.commonService.getInfo(),
      this.commonService.getSolved(true),
    ]);
    this.commonService.status.hasData = true;
  }
}
